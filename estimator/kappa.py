from enum import Enum

class KappaEnum(Enum):
    no = range(0,10)
    slight = range(10, 21)
    fair = range(21, 41)
    moderate = range(41, 61)
    substantial = range(61, 81)
    near_perfect = range(81, 100)
    perfect = range(100, 101)

def agree_no(k:float) -> bool:
    return int(k*100) in KappaEnum.no.value or (k <= 0)

def agree_slight(k:float) -> bool:
    return int(k*100) in KappaEnum.slight.value

def agree_fair(k:float) -> bool:
    return int(k*100) in KappaEnum.fair.value

def agree_moderate(k:float) -> bool:
    return int(k*100) in KappaEnum.moderate.value

def agree_subtantial(k:float) -> bool:
    return int(k*100) in KappaEnum.substantial.value

def agree_near_perfect(k:float) -> bool:
    return int(k*100) in KappaEnum.near_perfect.value

def agree_perfect(k:float) -> bool:
    return int(k*100) in KappaEnum.perfect.value

def agree_interpret(k:float) -> KappaEnum | None:
    if agree_no(k):
        return KappaEnum.no
    if agree_slight(k):
        return KappaEnum.slight
    if agree_fair(k):
        return KappaEnum.fair
    if agree_moderate(k):
        return KappaEnum.moderate
    if agree_subtantial(k):
        return KappaEnum.substantial
    if agree_near_perfect(k):
        return KappaEnum.near_perfect
    if agree_perfect(k):
        return KappaEnum.perfect
    return None
