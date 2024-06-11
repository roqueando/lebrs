from enum import StrEnum
from dataclasses import dataclass

STATS = {
    "hppool": "hp",
    "physicaldamage": "attackdamage",
    "magicdamage": "attackdamage",
    "armor": "armor",
    "attackspeed": "attackspeed",
    "spellblock": "spellblock",
    "movementspeed": "movespeed",
    "mppool": "mp",
    "critchance": "crit",
    "hpregen": "hpregen"
}

class EffectMod(StrEnum):
    ADD = 'add'
    TIMES = 'times'
    NOTHING = 'nothing'

@dataclass
class Effect:
    operator: EffectMod
    stat_field: str
    value: any

def get_item_effect(key: str, value) -> Effect:
    if 'Flat' in key:
        replaced = key.replace("Flat", "").replace("Mod", "")
        new_key = replaced.lower()
        stat_field = STATS[new_key]
        return Effect(operator=EffectMod.ADD, stat_field=stat_field, value=value)

    elif 'Percent' in key:
        replaced = key.replace("Percent", "").replace("Mod", "")
        new_key = replaced.lower()
        stat_field = STATS[new_key]
        return Effect(operator=EffectMod.TIMES, stat_field=stat_field, value=value)

    else:
        return Effect(operator=EffectMod.NOTHING, stat_field="", value=0)
