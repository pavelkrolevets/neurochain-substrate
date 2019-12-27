use hex;
use std::error::Error;
use std::io;
use std::process;
use ndarray::{array, Array2, Axis};

use std::num::ParseIntError;

fn read_csv() -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_reader(io::stdin());
    for result in rdr.records() {
        let record = result?;
        println!("{:?}", record);
    }
    Ok(())
}

pub struct NN_Model {
    Weights: Vec<u8>,
    Intercept: i64,
    LearningRate: i64,
    Loss: i64,
}

fn train_model_regression(model: &NN_Model, data: &Vec<u8>, Y: i64) -> Result<NN_Model, ParseIntError> {

    let mut error: i64 = 0;
    
    let mut prediction: i64 = model.Intercept;
    // Compute error
    for i in 0..model.Weights.len(){
        prediction += data[i] as i64 * model.Weights[i] as i64;
    }
    error = (prediction - Y).pow(2)/2 as i64;

    // Compute gradients and update weights
    let mut updated_weights: Vec<u8> = Vec::new();

    let mut w: i64 = 0;
    for i in 0..model.Weights.len(){
        w = w - ((prediction - Y) / model.LearningRate * data[i] as i64) as i64;
        updated_weights.push(w as u8);
    }
    
    // Compute gradients  and update intercept
    let mut updated_intercept: i64 = model.Intercept;
    updated_intercept -= ((prediction - Y) / model.LearningRate) as i64 ;

    //Compute new error
    let mut new_prediction: i64 = updated_intercept;
    for i in 0..updated_weights.len() {
        new_prediction +=  data[i] as i64 * updated_weights[i] as i64;
    }
    let new_loss = (new_prediction - Y).pow(2)/2 as i64;

    // Commit new model
    let updated_model = NN_Model{
        Weights: updated_weights,
        Loss: new_loss,
        Intercept: updated_intercept,
        LearningRate: model.LearningRate
    };

    Ok(updated_model)
}

fn predict (model: &NN_Model, data: &Vec<u8>) -> Result<i64, ParseIntError> {

    let mut m = model.Intercept ;
    for i in 0..model.Weights.len(){
        m = m + model.Weights[i] as i64 * data[i] as i64;
    }

    Ok(m)
}

fn main() {
    let a: Vec<u8>;
    let b: &[u8] = &[10u8, 20u8, 30u8, 10u8, 20u8, 30u8];
    let weight: Vec<u8> = b.iter().cloned().collect();
    let data: Vec<u8> = [1,1,2,3,3,3].to_vec();

    let mut result: i64 = 0;

    for i in 0..weight.len(){
        let mut new_weights: Vec<u8>;
        result = result + weight[i] as i64 * data[i] as i64;
    }
    let encoded = hex::encode(&weight);
    let decoded =  hex::decode(&encoded);

    println!{"Encoded weights {:?}", encoded};
    println!{"Encoded data {:?}", hex::encode(&data)};
    println!{"Converted {:?}", decoded};

    // let mut start_model = NN_Model{
    //     Weights: weight,
    //     Intercept: 10,
    //     LearningRate: 100,
    //     Loss: 0,
    // };
    // let Y: i64 = 200;
    // for i in 0..1000 {
    //     let mut model = train_model_regression(&start_model, &data, Y).unwrap();
    //     start_model = model;
    //     println!{"Loss: {:?}", start_model.Loss}
    //     println!{"Prediction: {:?}", predict(&start_model, &data)};
    // }

    // let a = arr2(&[[1., 2.],
    //     [0., 1.]]);
    // let b = arr2(&[[1., 2.],
    //     [2., 3.]]);

    // println!("Dot: {:?}", a.dot(&b));


   // READ MNIST DATASET TO ARRAY
   let mut X_train: Vec<Vec<f64>> = Vec::new();

   let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b',')
        .from_reader(io::stdin());
    for result in rdr.records() {
        let record = result.unwrap();
        // println!("{:?}", record);
        let mut record_as_vec: Vec<f64> = Vec::new();
        for field in record.iter(){
            &record_as_vec.push(field.parse().unwrap());
        }
        println!("{:?}", &record_as_vec);
        X_train.push(record_as_vec);
    }
    println!("{:?}", X_train);

}
