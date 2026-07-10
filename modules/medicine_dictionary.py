"""
medicine_dictionary.py
------------------------
A local, offline dictionary of common generic and brand medicine names
(global + South Asian / Pakistani market) used to fuzzy-match OCR text
against known drug names. This lets the OCR engine work completely
offline with no network calls or external AI APIs, while still giving
reasonable "is this actually a medicine name" confidence.

The list is intentionally broad rather than exhaustive — it is a
heuristic aid for the recognition pipeline, not a verified formulary.
Always confirm extracted medicine names with a licensed pharmacist.
"""

from __future__ import annotations

# Generic / international non-proprietary names
GENERIC_NAMES = [
    "Paracetamol", "Acetaminophen", "Ibuprofen", "Aspirin", "Amoxicillin",
    "Amoxicillin Clavulanate", "Azithromycin", "Ciprofloxacin", "Metronidazole",
    "Omeprazole", "Esomeprazole", "Pantoprazole", "Ranitidine", "Domperidone",
    "Metoclopramide", "Loperamide", "Ondansetron", "Cetirizine", "Loratadine",
    "Fexofenadine", "Chlorpheniramine", "Diphenhydramine", "Prednisolone",
    "Dexamethasone", "Hydrocortisone", "Metformin", "Glimepiride", "Gliclazide",
    "Insulin", "Atorvastatin", "Rosuvastatin", "Simvastatin", "Amlodipine",
    "Losartan", "Valsartan", "Telmisartan", "Atenolol", "Bisoprolol",
    "Metoprolol", "Enalapril", "Lisinopril", "Furosemide", "Hydrochlorothiazide",
    "Spironolactone", "Diclofenac", "Naproxen", "Mefenamic Acid", "Tramadol",
    "Codeine", "Morphine", "Gabapentin", "Pregabalin", "Amitriptyline",
    "Sertraline", "Fluoxetine", "Escitalopram", "Diazepam", "Alprazolam",
    "Clonazepam", "Levothyroxine", "Salbutamol", "Albuterol", "Montelukast",
    "Budesonide", "Fluticasone", "Cefixime", "Cefuroxime", "Cephalexin",
    "Doxycycline", "Clarithromycin", "Erythromycin", "Levofloxacin",
    "Fluconazole", "Clotrimazole", "Mupirocin", "Betamethasone", "Calamine",
    "Folic Acid", "Ferrous Sulfate", "Vitamin B Complex", "Vitamin C",
    "Vitamin D3", "Calcium Carbonate", "Multivitamin", "Zinc Sulfate",
    "Ondansetron", "Ranitidine", "Famotidine", "Lactulose", "Bisacodyl",
    "ORS", "Oral Rehydration Salts", "Insulin Glargine", "Insulin Aspart",
    "Warfarin", "Clopidogrel", "Rivaroxaban", "Digoxin", "Nitroglycerin",
    "Isosorbide Mononitrate", "Allopurinol", "Colchicine", "Methotrexate",
    "Hydroxychloroquine", "Prednisone", "Tamsulosin", "Finasteride",
    "Sildenafil", "Tadalafil", "Misoprostol", "Oxytocin", "Ergometrine",
]

# Common brand names sold in Pakistan / South Asia (and widely internationally)
BRAND_NAMES = [
    "Panadol", "Panadol Extra", "Calpol", "Brufen", "Disprin", "Augmentin",
    "Amoxil", "Zithromax", "Azomax", "Ciproxin", "Flagyl", "Risek",
    "Nexum", "Ppium", "Zantac", "Motilium", "Maxolon", "Imodium",
    "Zofran", "Zyrtec", "Claritin", "Telfast", "Avil", "Deltacortril",
    "Decadron", "Glucophage", "Amaryl", "Diamicron", "Lipitor", "Crestor",
    "Zocor", "Norvasc", "Cozaar", "Diovan", "Micardis", "Tenormin",
    "Concor", "Betaloc", "Renitec", "Zestril", "Lasix", "Hydrochlorothiazide",
    "Aldactone", "Voren", "Voltral", "Ponstan", "Tramal", "Ultracet",
    "Neurontin", "Lyrica", "Tryptanol", "Lustral", "Prozac", "Cipralex",
    "Valium", "Xanax", "Rivotril", "Eltroxin", "Ventolin", "Asthalin",
    "Singulair", "Pulmicort", "Flixotide", "Cefspan", "Zinnat", "Ceporex",
    "Vibramycin", "Klaricid", "Erythrocin", "Tavanic", "Diflucan", "Canesten",
    "Bactroban", "Betnovate", "Caladryl", "Folicid", "Ferro-Sanol",
    "Surbex-Z", "Cecon", "Osteocare", "Centrum", "Zincovit", "Duphalac",
    "Dulcolax", "Panadol CF", "Arinac", "Rigix", "Grip Away", "Flutec",
    "Actifed", "Corex", "Benadryl", "Augmex", "Klavox", "Moxatag",
    "Rivo", "Plavix", "Xarelto", "Lanoxin", "Angised", "Monocinque",
    "Zyloric", "Colcrys", "Methotrexate", "Plaquenil", "Flomax", "Proscar",
    "Viagra", "Cialis", "Cytotec", "Syntocinon", "Methergin",
]

ALL_MEDICINE_NAMES = sorted(set(GENERIC_NAMES + BRAND_NAMES))

# Common prescription abbreviations -> human-readable frequency, used to
# translate shorthand doctors write (Latin dosing abbreviations).
FREQUENCY_ABBREVIATIONS = {
    "OD": "once daily",
    "BD": "twice daily",
    "BID": "twice daily",
    "TDS": "three times daily",
    "TID": "three times daily",
    "QID": "four times daily",
    "QDS": "four times daily",
    "HS": "at bedtime",
    "SOS": "as needed",
    "PRN": "as needed",
    "STAT": "immediately",
    "AC": "before meals",
    "PC": "after meals",
    "OM": "every morning",
    "ON": "every night",
}
