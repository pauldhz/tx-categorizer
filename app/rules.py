from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple, List

def normalize_description(text: str) -> str:
    """Normalize bank descriptions to make matching more robust."""
    if text is None:
        return ""
    s = str(text).strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass(frozen=True)
class Rule:
    id: str
    pattern: re.Pattern
    category: str
    subcategory: str


# IMPORTANT: Order matters. First match wins.
RULES: List[Rule] = [
    Rule(id="R001", pattern=re.compile(r"^\s*PASS\s*$"), category="CHARGES_VARIABLES", subcategory="TRANSPORTS_COMMUN"),
    Rule(id="R002", pattern=re.compile(r"^\s*TOTAL\b.*"), category="CHARGES_VARIABLES", subcategory="CARBURANT"),
    Rule(id="R003", pattern=re.compile(r"Incoming\ transfer\ from\ M\ PAUL\ DENHEZ\ \(FR7616275500000412664270831\)"), category="DIVERS", subcategory="AJUSTEMENTS_ERREURS"),
    Rule(id="R004", pattern=re.compile(r"Paiement\ accepté:\ FR7616275500000412664270831\ à\ DE74502109007020623696"), category="DIVERS", subcategory="AJUSTEMENTS_ERREURS"),
    Rule(id="R005", pattern=re.compile(r"\bAMAZON\.FR\*[A-Z0-9]+"), category="ACHATS", subcategory="DIVERS"),
    Rule(id="R006", pattern=re.compile(r"Incoming\ transfer\ from\ Paul\ Denhez\ \(FR802043302626N269793476611\)"), category="DIVERS", subcategory="AJUSTEMENTS_ERREURS"),
    Rule(id="R007", pattern=re.compile(r"Incoming\ transfer\ from\ M\ PAUL\ DENHEZ"), category="DIVERS", subcategory="AJUSTEMENTS_ERREURS"),
    Rule(id="R008", pattern=re.compile(r"GARFO\ \-\ FOOD\ \&\ BEVERAGE\."), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R009", pattern=re.compile(r"CARRIS\ \-\ RUA\ 1\ MAIO,\-00"), category="ACHATS", subcategory="DIVERS"),
    Rule(id="R010", pattern=re.compile(r"NESPRESSO\ FRANCE\ S\.A\.S\."), category="ACHATS", subcategory="CAFE"),
    Rule(id="R011", pattern=re.compile(r"LS\ LA\ COUR\ DE\ LA\ CHTI"), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R012", pattern=re.compile(r"SNC\ LE\ BIENVENU\ 4069410"), category="ACHATS", subcategory="TABAC"),
    Rule(id="R013", pattern=re.compile(r"Metropolitano\ de\ Lisboa"), category="CHARGES_VARIABLES", subcategory="TRANSPORTS_COMMUN"),
    Rule(id="R014", pattern=re.compile(r"NYX\*LILLEAUTOMATIQUEDIST"), category="ACHATS", subcategory="CAFE"),
    Rule(id="R015", pattern=re.compile(r"PHAR\ BOURGMAYER\ 4194069"), category="SANTE", subcategory="PHARMACIE"),
    Rule(id="R016", pattern=re.compile(r"Association\ Ruban\ Rose"), category="DIVERS", subcategory="DONS"),
    Rule(id="R017", pattern=re.compile(r"Cash\ reward\ allocation"), category="DIVERS", subcategory="AJUSTEMENTS_ERREURS"),
    Rule(id="R018", pattern=re.compile(r"NYX\*VALENCIENNESPLACEDA"), category="ACHATS", subcategory="DIVERS"),
    Rule(id="R019", pattern=re.compile(r"PHARMACIE\ VALS\ 2151306"), category="SANTE", subcategory="AUTRE_MEDECINE"),
    Rule(id="R020", pattern=re.compile(r"PICARD\ SA\ 335\ 4998985"), category="ALIMENTATION", subcategory="COURSES"),
    Rule(id="R021", pattern=re.compile(r"BAR\ BILTOKI\ HALLES\ D"), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R022", pattern=re.compile(r"CONTIN\ BOM\ DIA\ LISBO"), category="ALIMENTATION", subcategory="COURSES"),
    Rule(id="R023", pattern=re.compile(r"Your\ Saveback\ payment"), category="DIVERS", subcategory="AJUSTEMENTS_ERREURS"),
    Rule(id="R024", pattern=re.compile(r"ELECTRO\ DEPOT\ FRANCE"), category="MAISON", subcategory="EQUIPEMENT_ELECTROMENAGER"),
    Rule(id="R025", pattern=re.compile(r"LISBON\ DUTY\ FREE\ T2"), category="LOISIRS", subcategory="VACANCES_WEEKENDS"),
    Rule(id="R026", pattern=re.compile(r"CIVETTE\ DE\ LA\ TOUR"), category="ACHATS", subcategory="TABAC"),
    Rule(id="R027", pattern=re.compile(r"MIGUEL\ CASTRO\-SILVA"), category="ACHATS", subcategory="DIVERS"),
    Rule(id="R028", pattern=re.compile(r"PADEL\ FOOTBALL\ CLUB"), category="LOISIRS", subcategory="SPORT"),
    Rule(id="R029", pattern=re.compile(r"RELAY\ TRIBS\ 4116230"), category="ACHATS", subcategory="DIVERS"),
    Rule(id="R030", pattern=re.compile(r"RESTAURANTE\ FERNANDO"), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R031", pattern=re.compile(r"4PADEL\ Valenciennes"), category="LOISIRS", subcategory="SPORT"),
    Rule(id="R032", pattern=re.compile(r"CONTINENTE\ BOM\ DIA"), category="ACHATS", subcategory="DIVERS"),
    Rule(id="R033", pattern=re.compile(r"COURIR\ VALENCIENNES"), category="ACHATS", subcategory="VETEMENTS"),
    Rule(id="R034", pattern=re.compile(r"FERME\ DU\ PONT\ DES"), category="ALIMENTATION", subcategory="COURSES"),
    Rule(id="R035", pattern=re.compile(r"GRAND\ FRAIS\ AULNOY"), category="ALIMENTATION", subcategory="COURSES"),
    Rule(id="R036", pattern=re.compile(r"MCDONALDS\ AEROPORTO"), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R037", pattern=re.compile(r"MGP\*Le\ Pot\ Commun"), category="ACHATS", subcategory="CADEAUX"),
    Rule(id="R038", pattern=re.compile(r"GELATOMANIA\ NAZARE"), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R039", pattern=re.compile(r"LE\ CYRANO\ 4266161"), category="ACHATS", subcategory="TABAC"),
    Rule(id="R040", pattern=re.compile(r"LE\ JUBILE\ 4357453"), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R041", pattern=re.compile(r"BIE\ DE\ LA\ HALLE"), category="ALIMENTATION", subcategory="BOUCHERIE"),
    Rule(id="R042", pattern=re.compile(r"E0022API\ EDS\ ONE"), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R043", pattern=re.compile(r"PADEL\ FOOTBALL\ C"), category="LOISIRS", subcategory="SPORT"),
    Rule(id="R044", pattern=re.compile(r"PASTEIS\ DE\ BELEM"), category="ALIMENTATION", subcategory="COURSES"),
    Rule(id="R045", pattern=re.compile(r"SHIFU\ RAMEN\ REST"), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R046", pattern=re.compile(r"SumUp\ \*SBCONCEPT"), category="ACHATS", subcategory="SOIN DE LA PERSONNE"),
    Rule(id="R047", pattern=re.compile(r"TERRACO\ EDITORIAL"), category="ACHATS", subcategory="DIVERS"),
    Rule(id="R048", pattern=re.compile(r"Interest\ payment"), category="BANQUE", subcategory="INTERETS"),
    Rule(id="R049", pattern=re.compile(r"MCDONALDS\ CHIADO"), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R050", pattern=re.compile(r"SnP\*SPEED\ PIZZA"), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R051", pattern=re.compile(r"BEER\ EXPERIENCE"), category="ACHATS", subcategory="DIVERS"),
    Rule(id="R052", pattern=re.compile(r"CHEZ\ MON\ VIEUX"), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R053", pattern=re.compile(r"DCTR\ MARGUERITT"), category="SANTE", subcategory="MEDECIN"),
    Rule(id="R054", pattern=re.compile(r"GD\ FRAIS\ SENTI"), category="ALIMENTATION", subcategory="COURSES"),
    Rule(id="R055", pattern=re.compile(r"MA\ DUQUE\ LOULE"), category="ACHATS", subcategory="DIVERS"),
    Rule(id="R056", pattern=re.compile(r"SINTRA\ LRO\ TVM"), category="CHARGES_VARIABLES", subcategory="TRANSPORTS_COMMUN"),
    Rule(id="R057", pattern=re.compile(r"APPLE\.COM/BILL"), category="CHARGES_FIXES", subcategory="ABONNEMENTS_FIXES"),
    Rule(id="R058", pattern=re.compile(r"DELEBARRE\ VINS"), category="ACHATS", subcategory="CADEAUX"),
    Rule(id="R059", pattern=re.compile(r"EURL\ A\ MOREAU"), category="ALIMENTATION", subcategory="BOULANGERIE"),
    Rule(id="R060", pattern=re.compile(r"FERME\ DU\ SART"), category="ALIMENTATION", subcategory="COURSES"),
    Rule(id="R061", pattern=re.compile(r"M'MA\ TURINETTI"), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R062", pattern=re.compile(r"MARIE\ BLACHERE"), category="ALIMENTATION", subcategory="COURSES"),
    Rule(id="R063", pattern=re.compile(r"SINTRA\ TERRACE"), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R064", pattern=re.compile(r"TIME\ OUT\ SHOP"), category="LOISIRS", subcategory="VACANCES_WEEKENDS"),
    Rule(id="R065", pattern=re.compile(r"VINS\ GOURMANDS"), category="ACHATS", subcategory="VIN"),
    Rule(id="R066", pattern=re.compile(r"WEB\ TENNIS\ SC"), category="LOISIRS", subcategory="SPORT"),
    Rule(id="R067", pattern=re.compile(r"SUR\ LE\ POUCE"), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R068", pattern=re.compile(r"Zettle_\*Sahil"), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R069", pattern=re.compile(r"LE\ LONGCHAMP"), category="ACHATS", subcategory="TABAC"),
    Rule(id="R070", pattern=re.compile(r"MAISON\ RINC"), category="ACHATS", subcategory="CADEAUX"),
    Rule(id="R071", pattern=re.compile(r"SAS\ BONDUWE"), category="ACHATS", subcategory="DIVERS"),
    Rule(id="R072", pattern=re.compile(r"SPEED\ PIZZA"), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R073", pattern=re.compile(r"TCE\ 4332548"), category="LOISIRS", subcategory="SPORT"),
    Rule(id="R074", pattern=re.compile(r"VAL\ VIANDES"), category="ALIMENTATION", subcategory="BOUCHERIE"),
    Rule(id="R075", pattern=re.compile(r"CAFES\ REMY"), category="ACHATS", subcategory="CAFE"),
    Rule(id="R076", pattern=re.compile(r"INTERMARCHE"), category="ALIMENTATION", subcategory="COURSES"),
    Rule(id="R077", pattern=re.compile(r"LE\ VALENCY"), category="ACHATS", subcategory="TABAC"),
    Rule(id="R078", pattern=re.compile(r"VINI\ LILLE"), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R079", pattern=re.compile(r"SP\ WILDDE"), category="ACHATS", subcategory="SOIN DE LA PERSONNE"),
    Rule(id="R080", pattern=re.compile(r"RIGOLETTO"), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R081", pattern=re.compile(r"SAS\ BETA"), category="ALIMENTATION", subcategory="BOULANGERIE"),
    Rule(id="R082", pattern=re.compile(r"EL\ GANA"), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R083", pattern=re.compile(r"O\ TERA"), category="ALIMENTATION", subcategory="COURSES"),
    Rule(id="R084", pattern=re.compile(r"CHOOSE"), category="ACHATS", subcategory="DIVERS"),
    Rule(id="R085", pattern=re.compile(r"MYTHOS"), category="ALIMENTATION", subcategory="RESTAURANTS"),
    Rule(id="R086", pattern=re.compile(r"OXYBUL"), category="ACHATS", subcategory="CADEAUX"),
    Rule(id="R087", pattern=re.compile(r"VPC"), category="ACHATS", subcategory="CAFE"),
    Rule(id="R088", pattern=re.compile(r"\bSAVINGS PLAN EXECUTION\b.*"), category="EPARGNE", subcategory="INVESTISSEMENTS"),
    Rule(id="R089", pattern=re.compile(r"\bAMAZON PAYMENTS\b.*"), category="ACHATS", subcategory="DIVERS"),
    Rule(id="R090", pattern=re.compile(r"\bALIM CARREFOUR\b.*"), category="ALIMENTATION", subcategory="COURSES"),
    Rule(id="R091", pattern=re.compile(r"\bAMAZON EU SARL\b.*"), category="ACHATS", subcategory="DIVERS"),
    Rule(id="R092", pattern=re.compile(r"\bAMAZON PRIME\b.*"), category="CHARGES_FIXES", subcategory="ABONNEMENTS_FIXES"),
    Rule(id="R093", pattern=re.compile(r"\bLEROY MERLIN\b.*"), category="MAISON", subcategory="BRICOLAGE"),
    Rule(id="R094", pattern=re.compile(r"\bAMZN MKTP\b.*"), category="ACHATS", subcategory="DIVERS"),
    Rule(id="R095", pattern=re.compile(r"\bZENPARK\b.*"), category="CHARGES_VARIABLES", subcategory="STATIONNEMENT_PEAGES"),
]

def predict_by_rules(description: str) -> Optional[Tuple[str, str, str]]:
    """
    Returns (category, subcategory, rule_id) if a rule matches, else None.
    """
    d = normalize_description(description)
    for rule in RULES:
        if rule.pattern.search(d):
            return (rule.category, rule.subcategory, rule.id)
    return None


if __name__ == "__main__":
    # Quick manual smoke test
    tests = [
        "Apple.com/bill",
        "AMAZON PRIME FR 2469664",
        "Amazon.fr*ZX9NA3KU4",
        "ZENPARKMAILLERI 4030783",
        "PASS",
        "Interest payment",
    ]
    for t in tests:
        print(t, "->", predict_by_rules(t))
