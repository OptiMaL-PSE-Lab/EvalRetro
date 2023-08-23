import itertools
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from collections import namedtuple

import pandas as pd
from pathlib import Path

import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog('rdApp.error')

data_path = Path(__file__).parent / 'data'

AtomInfo = namedtuple('AtomInfo',('mapnum','reactant','reactantAtom','product','productAtom'))

# Taken from Greg Landrum's code on extracting reaction information

def map_reacting_atoms_to_products(rxn,reactingAtoms):
    ''' 
    figures out which atoms in the products each mapped atom in the reactants maps to 
    Returns a namedtuple with reactant info and product info 
    '''
    res = []
    for ridx,reacting in enumerate(reactingAtoms):
        reactant = rxn.GetReactantTemplate(ridx)
        for raidx in reacting:
            mapnum = reactant.GetAtomWithIdx(raidx).GetAtomMapNum()
            foundit=False
            for pidx,product in enumerate(rxn.GetProducts()):
                for paidx,patom in enumerate(product.GetAtoms()):
                    if patom.GetAtomMapNum()==mapnum:
                        res.append(AtomInfo(mapnum,ridx,raidx,pidx,paidx))
                        foundit = True
                        break
                    if foundit:
                        break
    return res
def get_mapped_neighbors(atom):
    ''' 
    test all mapped neighbors of a mapped atom
    Extracts atom indeces for neighbouring pairs
    '''
    res = {}
    amap = atom.GetAtomMapNum()
    if not amap:
        return res
    for nbr in atom.GetNeighbors():
        nmap = nbr.GetAtomMapNum()
        if nmap:
            if amap>nmap:
                res[(nmap,amap)] = (atom.GetIdx(),nbr.GetIdx())
            else:
                res[(amap,nmap)] = (nbr.GetIdx(),atom.GetIdx())
    return res

BondInfo = namedtuple('BondInfo',('product','productAtoms','productBond','status'))

def find_modifications_in_products(rxn):
    ''' returns a 2-tuple with the modified atoms and bonds from the reaction '''
    reactingAtoms = rxn.GetReactingAtoms()
    amap = map_reacting_atoms_to_products(rxn,reactingAtoms)
    res = []
    seen = set()
    # this is all driven from the list of reacting atoms:
    for _,ridx,raidx,pidx,paidx in amap:
        reactant = rxn.GetReactantTemplate(ridx)
        ratom = reactant.GetAtomWithIdx(raidx)
        product = rxn.GetProductTemplate(pidx)
        patom = product.GetAtomWithIdx(paidx)

        rnbrs = get_mapped_neighbors(ratom)
        pnbrs = get_mapped_neighbors(patom)
        for tpl in pnbrs:
            pbond = product.GetBondBetweenAtoms(*pnbrs[tpl])
            if (pidx,pbond.GetIdx()) in seen:
                continue
            seen.add((pidx,pbond.GetIdx()))
            if not tpl in rnbrs:
                # new bond in product
                res.append(BondInfo(pidx,pnbrs[tpl],pbond.GetIdx(),'New'))
            else:
                # present in both reactants and products, check to see if it changed
                rbond = reactant.GetBondBetweenAtoms(*rnbrs[tpl])
                if rbond.GetBondType()!=pbond.GetBondType():
                    # Would refer to bond type changed -> not of interest to us
                    #! Investigate impact of combining the two cases
                    res.append(BondInfo(pidx,pnbrs[tpl],pbond.GetIdx(),'Changed'))
    return amap,res


def create_training_labels(rxn_smi):
    """ 
    Create training labels (y) for changing bonds and atoms  
    """
    rxn = rdChemReactions.ReactionFromSmarts(rxn_smi, useSmiles=True)
    rxn.Initialize()

    named_tuple = namedtuple('ReactionInfo',('rxn_bonds', 'rxn_atoms', 'y_bonds', 'y_atoms', 'rxnsmile'))
    amap, res = find_modifications_in_products(rxn)
    product = rxn.GetProductTemplate(0)
    len_atoms = len(product.GetAtoms())
    len_bonds = len(product.GetBonds())
    y_atoms = [0] * len_atoms
    y_bonds = [0] * len_bonds
    bonds = []
    for bond in res:
        bonds.append(bond.productBond)
        y_bonds[bond.productBond] = 1

    # Check for atoms where neither new bond changed nor formed/broken
    atoms_in_reactive_bonds = set([res[i].productAtoms for i in range(len(res))])
    atoms_in_reactive_bonds = list(itertools.chain.from_iterable(atoms_in_reactive_bonds))
    atoms = []
    for atom in amap:
        if atom.productAtom not in atoms_in_reactive_bonds:
            y_atoms[atom.productAtom] = 1
            atoms.append(atom.productAtom)
    
    return named_tuple(bonds, atoms, y_bonds, y_atoms, rxn_smi)


if __name__ == '__main__':
    example = pd.read_csv(data_path / 'canonicalized_test.csv')
    rxn = example['rxn'][2]
    print(create_training_labels(rxn))
