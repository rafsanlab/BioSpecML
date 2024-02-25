"""
Based from:

Meade, A.D., Lyng, F.M., Knief, P. et al. Growth substrate induced functional changes elucidated
by FTIR and Raman spectroscopy in in–vitro cultured human keratinocytes. Anal Bioanal Chem 387, 1717–1728 (2007).
https://doi.org/10.1007/s00216-006-0876-5 (Download link: https://arrow.tudublin.ie/radart/6/)

"""

wavenumbers_dict = {
    3070: 'Amide B (CNH bend)',
    2960: 'CH3 stretch (antisymmetric) due to methyl terminal of membrane phospholipids',
    2936: 'CH3 stretch',
    2928: 'CH2 antisymmetric stretch of Methylene group of membrane phospholipids',
    2886: 'CH2 stretch (symmetric) due to methylene groups of membrane phospholipids',
    2854: 'CH2 stretch',
    2739: 'CH stretch',
    1736: 'C=O stretch',
    (1667, 1640) : 'Amide I (protein) C=O stretching of amide coupled to NH2 in-plane bending',
    1657: 'C=C stretch (lipids), Amide I (α-helix, protein)',
    1659: 'C=C stretch (lipids), Amide I (α-helix, protein)',
    1611: 'Tyr (aromatics)',
    1566: 'Phe, Trp (phenyl, aromatics)',
    1550: 'Amide II absorption due to N-H bending coupled to a C-N stretch',
    1509: 'C=C stretch (aromatics)',
    1452: 'CH2 stretch deformation of methylene group (lipids)',
    1439: 'CH2 def.',
    1420: 'CH3 asymmetric stretch (lipids, aromatics)',
    1397: 'CH3 bending due to methyl bond in the membrane',
    1382: 'COO- symmetric stretch',
    1367: 'CH3 symmetric stretch',
    1336: 'Adenine, Phenylalanine, CH deformation',
    1304: 'Lipids CH2 twist, protein amide III band, adenine, cytosine',
    1267: 'Amide III (α-helix, protein)',
    1250: 'Amide III (β-sheet, protein)',
    1235: 'Antisymmetric phosphate stretching',
    1206: 'C-C stretch, C-H bend',
    1165: 'C-O stretch, COH bend',
    1130: 'C-C asymmetric stretch',
    1100: 'PO2- symmetric stretch (nucleic acids)',
    1094: 'PO2- symmetric stretch (nucleic acids)',
    1081: 'PO2- symmetric stretch (nucleic acids)',
    1065: 'Chain C-C',
    1056: 'RNA ribose C-O vibration',
    1003: 'Phenylalanine (ring-breathing)',
    967: 'C-C and C-N stretch PO3 2- stretch (DNA)',
    957: 'CH3 deformation (lipid, protein)',
    936: 'C-C residue α-helix',
    921: 'C-C stretch proline',
    898: 'C-C stretch residue',
    870: 'C-DNA',
    853: 'Ring breathing Tyr – C-C stretch proline',
    (828, 833) : 'Out of plane breathing Tyr; PO2- asymmetric stretch DNA (B-form)',
    807: 'A-DNA',
    786: 'DNA – RNA (PO2-) symmetric stretching',
    746: 'Thymine',
    727: 'Adenine'
}

wavenumbers_dict_simplified = {
    3070: 'δ(CNH) (Amide B)',
    2960: 'νₐₛ(CH₃)', # due to methyl terminal of membrane phospholipids
    2936: 'v(CH₃)',
    2928: 'vₐₛ(CH)₂', # of methylene group of membrane phospholipids'
    2886: 'vₛ(CH₂),', # of methylene groups of membrane phospholipids',
    2854: 'v(CH₂)',
    2739: 'v(CH)',
    1736: 'v(C=O)',
    (1667, 1640): 'v(C=O) Amide I', # , protein of coupled to NH2 in-plane bending
    1657: 'v(C=C) lipids, Amide I (α-helix)',
    1659: 'v(C=C) lipids, Amide I (α-helix)',
    1611: 'Tyr (aromatics)',
    1566: 'Phe, Trp (phenyl, aromatics)',
    1550: 'Amide II', # absorption due to N-H bending coupled to a C-N stretch
    1509: 'ν(C=C) aromatics',
    1452: 'νδ(CH₂) lipids methylene group', # RECHECK
    1439: 'δ(CH₂)',
    1420: 'vₐₛ(CH₃) lipids, aromatics',
    1397: 'δ(CH₃)',
    1382: 'vₛ(COO⁻)',
    1367: 'vₛ(CH₃)',
    1336: 'Adenine, Phe, δ(CH)',
    1304: 'twist (CH₂) lipids, amide III, adenine, cytosine',
    1267: 'Amide III (α-helix, protein)',
    1250: 'Amide III, (β-sheet, protein)',
    1235: 'vₐₛ(phosphate)',
    1206: 'v(C-C), δ(C-H)',
    1165: 'v(C-O), δ(COH)',
    1130: 'vₐₛ(C-C)',
    1100: 'vₛ(PO₂⁻) nucleic acids',
    1094: 'vₛ(PO₂⁻) nucleic acids',
    1081: 'vₛ(PO₂⁻) nucleic acids',
    1065: 'Chain C-C',
    1056: 'RNA ribose C-O vibration',
    1003: 'Phe (ring-breathing)',
    967: 'v(C-C), v(C-N), v(PO₃²⁻) DNA',
    957: 'δ(CH₃) lipid, protein',
    936: 'C-C residue (α-helix)',
    921: 'v(C-C) proline',
    898: 'v(C-C) residue',
    870: 'C-DNA',
    853: 'Ring breathing Tyr-(C-C) stretch proline',
    (828, 833): 'Out of plane breathing Tyr; vₐₛ(PO₂⁻) DNA B-form DNA',
    807: 'A-DNA',
    786: 'vₛ(PO₂⁻) DNA-RNA',
    746: 'Thymine',
    727: 'Adenine'
}