use candle_core::Tensor;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

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

#[derive(Serialize, Deserialize)]
pub struct BaseItem {
    stats: HashMap<String, i32>
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
    #[test]
    fn build_item() {
    }
}
