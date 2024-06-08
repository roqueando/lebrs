use candle_core::Tensor;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum Number {
    Integer(i32),
    Float(f32),
    Null
}

pub enum EffectMod {
    Add,
    Times
}

pub struct Effect {
    pub operator: EffectMod,
    pub stat_field: String
}

pub struct Item {
    pub effect: Effect
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BaseItem {
    pub id: String,
    pub name: String,
    pub stats: Option<HashMap<String, Number>>
}

impl BaseItem {
}

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
    pub attackspeed: f32
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
            attackspeed: 1.0
        }
    }

    pub fn update(&self, _items: Vec<Item>) -> Tensor {
        // TODO: apply all the item stats modifiers
        todo!()
    }
}

#[cfg(test)]
mod resources_test {
    use std::fs::File;
    use std::io::prelude::*;
    use super::*;
    //use polars::prelude::*;
    //use polars::lazy::dsl::col;

    #[test]
    fn build_item() {
        let mut file = File::open("data/item_processed.json").unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        let json: Vec<BaseItem> = serde_json::from_str(&contents).unwrap();
        println!("{:?}", json);
        assert_eq!(1 + 1, 2)
    }
}
