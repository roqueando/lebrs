use candle_core::Device;
use candle_core::Error;
use candle_core::Tensor;
use candle_nn::Optimizer;
use candle_nn::VarMap;
use candle_nn::SGD;
use candle_nn::{Linear, Module, VarBuilder};

pub enum ModelError {
    ModelNotTrainedEnough,
}

#[derive(Clone)]
pub struct Dataset {
    pub x_train: Tensor,
    pub x_test: Tensor,
    pub y_train: Tensor,
    pub y_test: Tensor,
}

pub struct LeBRS {
    pub inputs: Linear,
    pub hidden1: Linear,
    pub hidden2: Linear,
    pub output: Linear,
}

impl LeBRS {
    fn new(vb: VarBuilder) -> Result<Self, Error> {
        let inputs = candle_nn::linear(6, 8, vb.pp("inputs"))?;
        let hidden1 = candle_nn::linear(8, 8, vb.pp("hidden1"))?;
        let hidden2 = candle_nn::linear(8, 8, vb.pp("hidden2"))?;
        let output = candle_nn::linear(8, 6, vb.pp("ouptut"))?;

        Ok(Self {
            inputs,
            hidden1,
            hidden2,
            output,
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor, Error> {
        let xs = self.inputs.forward(xs)?;
        // TODO: change this relu for existent_item function
        // existent_item must return a tensor
        let xs = xs.relu()?;
        let xs = self.hidden1.forward(&xs)?;
        let xs = xs.relu()?;
        let xs = self.hidden2.forward(&xs)?;
        let xs = xs.relu()?;

        self.output.forward(&xs)
    }
}

pub fn train(data: Dataset, dev: &Device) -> Result<LeBRS, Error> {
    let x_train = data.x_train.to_device(dev)?;
    let _x_test = data.x_test.to_device(dev)?;
    let _y_train = data.y_train.to_device(dev)?;
    let _y_test = data.y_test.to_device(dev)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, dev);
    let model = LeBRS::new(vb.clone())?;

    // optimizer
    let mut _sgd = SGD::new(varmap.all_vars(), 0.005)?;
    let final_acc: f32 = 0.0;
    let out = model.forward(&x_train)?;
    println!("result: {}", out);

    /*
    for _epoch in 1..11 {
        // TODO: get items data, create a base_one_champion
        // calculate loss by mse
        // use sgd to optimize weights
    }
    */

    Ok(model)
    /*
    if final_acc == 100.0 {
        Err(candle_core::error::Error::Msg(
            "Model not trained enough".to_string(),
        ))
    } else {
        Ok(model)
    }
    */
}

#[cfg(test)]
mod neural_tests {
    #[test]
    fn training_test() {
        assert_eq!(1 + 1, 2)
    }
}
