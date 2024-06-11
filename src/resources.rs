use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum Number {
    Integer(i32),
    Float(f32),
    Null,
}

#[derive(Debug, PartialEq)]
pub enum EffectMod {
    Add,
    Times,
    Nothing,
}

#[derive(Debug, PartialEq)]
pub struct Effect {
    pub operator: EffectMod,
    pub stat_field: String,
    pub value: Number
}

#[derive(Debug)]
pub enum EffectError {
    EffectNotMatch,
}

#[derive(Debug, PartialEq)]
pub struct Item {
    pub id: String,
    pub effect: Effect,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BaseItem {
    pub id: String,
    pub name: String,
    pub stats: Option<HashMap<String, Number>>,
}

impl BaseItem {}

#[derive(Clone)]
pub struct BaseOneChampion {
    pub hp: i32,
    pub mp: i32,
    pub movespeed: i32,
    pub armor: i32,
    pub spellblock: i32,
    pub attackrange: i32,
    pub hpregen: i32,
    pub mpregen: i32,
    pub crit: i32,
    pub attackdamage: i32,
    pub attackspeed: f32,
}

impl BaseOneChampion {
    pub fn build() -> Self {
        Self {
            hp: 1,
            mp: 1,
            movespeed: 1,
            armor: 1,
            spellblock: 1,
            attackrange: 1,
            hpregen: 1,
            mpregen: 1,
            crit: 1,
            attackdamage: 1,
            attackspeed: 1.0,
        }
    }

    pub fn update(&self, _items: Vec<Item>) -> Tensor {
        // TODO: apply all the item stats modifiers
        todo!()
    }
}

pub fn load_items(path: &str) -> Vec<BaseItem> {
    let mut file = File::open(path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    let items: Vec<BaseItem> = serde_json::from_str(&contents).unwrap();
    return items;
}

pub fn stats_from_to() -> HashMap<&'static str, &'static str> {
    let mut stats = HashMap::new();

    stats.insert("hppool", "hp");
    stats.insert("physicaldamage", "attackdamage");
    stats.insert("magicdamage", "attackdamage");
    stats.insert("armor", "armor");
    stats.insert("attackspeed", "attackspeed");
    stats.insert("spellblock", "spellblock");
    stats.insert("movementspeed", "movespeed");
    stats.insert("mppool", "mp");
    stats.insert("critchance", "crit");
    stats.insert("hpregen", "hpregen");

    return stats;
}

pub fn get_item_effect(key: &str, value: Number) -> Effect {
    let stats_base = stats_from_to();
    if key.contains("Flat") {
        let replaced = key.replace("Flat", "");
        let replaced = replaced.replace("Mod", "");
        let new_key = replaced.to_lowercase();

        if let Some(field) = stats_base.get(new_key.as_str()) {
            Effect {
                operator: EffectMod::Add,
                stat_field: field.to_string(),
                value
            }
        } else {
            Effect {
                operator: EffectMod::Nothing,
                stat_field: "".to_string(),
                value
            }
        }
    } else {
        let replaced = key.replace("Percent", "");
        let replaced = replaced.replace("Mod", "");
        let new_key = replaced.to_lowercase();

        if let Some(field) = stats_base.get(new_key.as_str()) {
            Effect {
                operator: EffectMod::Times,
                stat_field: field.to_string(),
                value
            }
        } else {
            Effect {
                operator: EffectMod::Nothing,
                stat_field: "".to_string(),
                value
            }
        }
    }
}

pub fn process_items(items: Vec<BaseItem>) -> Vec<Item> {
    let mut new_items = vec![];

    for item in items.into_iter() {
        if let Some(stat) = &item.stats {
            let keys = stat.keys();

            for key in keys {
                if let Some(value) = stat.get(key) {
                    let effect = get_item_effect(key, value.clone());
                    let id = &item.id;
                    new_items.push(Item { id: id.to_string(), effect })
                }
            }
        }
    }

    return new_items;
}

#[cfg(test)]
mod resources_test {
    use super::*;

    #[test]
    fn build_item() {
        let items = load_items("data/item_processed.json");
        let processed = process_items(items);

        assert_eq!(processed[0],
            Item {
                id: "1001".to_string(),
                effect: Effect {
                    operator: EffectMod::Add,
                    stat_field: "movespeed".to_string(),
                    value: Number::Integer(25)
                }
            })
    }

    #[test]
    fn calculate_stats() {
        let items = load_items("data/item_processed.json");
        let processed = process_items(items);
        let chosen_items: Vec<&str> = vec!["1054", "3047", "3133", "3057", "3067"];
        //let chosen_processed_items = processed.into_iter()
        //    .filter(|item| {
        //        for ci in chosen_items {
        //            return item.id.as_str() == ci 
        //        }
        //        //&item.id.as_str() == chosen_items
        //    });
        
        // TODO: probably I'll need to zip with chosen_items and filter with these two ids

        //let base_one = BaseOneChampion::build()
        //    .update(chosen_processed_items);

        //println!("{:?}", processed.into_iter().find(|item: &Item| item.id == "1001".to_string()));
        //println!("{:?}", chosen_processed_items);

    }
}
