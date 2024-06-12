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
    "hpregen": "hpregen",
}

STAT_INDEX = {
    'hp': 0,
    'mp': 1,
    'movespeed': 2,
    'armor': 3,
    'spellblock': 4,
    'attackrange': 5,
    'hpregen': 6,
    'mpregen': 7,
    'crit': 8,
    'attackdamage': 9,
    'attackspeed': 10
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
        if new_key == 'lifesteal':
            return Effect(operator=EffectMod.NOTHING, stat_field="", value=0)

        stat_field = STATS[new_key]
        return Effect(operator=EffectMod.ADD, stat_field=stat_field, value=value)

    elif 'Percent' in key:
        replaced = key.replace("Percent", "").replace("Mod", "")
        new_key = replaced.lower()
        if new_key == 'lifesteal':
            return Effect(operator=EffectMod.NOTHING, stat_field="", value=0)

        stat_field = STATS[new_key]
        return Effect(operator=EffectMod.TIMES, stat_field=stat_field, value=value)

    else:
        return Effect(operator=EffectMod.NOTHING, stat_field="", value=0)


def update_stats(boc: dict, items: list[item_calculator.Effect]) -> list:
    updated_boc = {**boc}
    for item in items:
        if item.operator == item_calculator.EffectMod.ADD:
            updated_boc[item.stat_field] = boc[item.stat_field] + item.value
        if item.operator == item_calculator.EffectMod.TIMES:
            updated_boc[item.stat_field] = boc[item.stat_field] * item.value

    return list(updated_boc.values())


def choose_items(items: list, df: DataFrame) -> list[item_calculator.Effect]:
    effects = []
    for item in items:
        item_stats = df.loc[df['id'] == item]['stats']
        if len(item_stats.values) > 0:
            if item_stats.values[0] is not None:
                for key, value in item_stats.values[0].items():
                    effects.append(item_calculator.get_item_effect(key, value))
    return effects
