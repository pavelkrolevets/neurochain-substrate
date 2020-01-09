use hex;
use std::error::Error;
use std::io;
use std::process;
use ndarray::{array, Array2, Axis};

use std::num::ParseIntError;

extern crate mnist;
extern crate rulinalg;
extern crate rand;
extern crate rand_distr;

use rand::distributions::{Normal, Distribution};
use mnist::{Mnist, MnistBuilder};
use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

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


//   // READ MNIST DATASET TO ARRAY
//   let mut X_train: Vec<Vec<f64>> = Vec::new();
//
//   let mut rdr = csv::ReaderBuilder::new()
//        .delimiter(b',')
//        .from_reader(io::stdin());
//    let mut counter = 0;
//    for result in rdr.records() {
//        counter += 1;
//        let record = result.unwrap();
//        // println!("{:?}", record);
//        let mut record_as_vec: Vec<f64> = Vec::new();
//        for field in record.iter(){
//          let mut val: f64 = field.parse().unwrap();
//            val /= 255f64;
//            &record_as_vec.push(val);
//        }
//        println!("Parsing record {:?}", counter);
//        X_train.push(record_as_vec);
//
//        if counter == 10000 {
//            break
//        }
//    }
//    println!(" Finished parsing X_train");

    let (trn_size, rows, cols) = (50_000, 28, 28);
    // Deconstruct the returned Mnist struct.
    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    // Get the label of the first digit.
    let first_label = trn_lbl[0];
    println!("The first digit is a {}.", first_label);

    //Model parameters
    let layer_size1 = 10;
    let layer_size2 = 10;
    let K = 10;



    // Convert the flattened training images vector to a matrix.
    let trn_img = Matrix::new(trn_size as usize, (cols * rows) as usize, trn_img);

    ////////////////////////////////
    // Initialize weights and biases
    ////////////////////////////////

    // Random vector for weight initialization mean 0, standard deviation 1
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let mut w1_v: Vec<f64> = vec![0.0; (cols * rows) * layer_size1];
    for x in w1_v.iter_mut() {
        *x = normal.sample(&mut rand::thread_rng())
    }
    let W1 = Matrix::new((cols * rows) as usize, layer_size1 as usize, w1_v);

    let normal = Normal::new(0.0, 1.0);
    let mut b1_v: Vec<f64> = vec![0.0; (cols * rows) * layer_size1];
    for x in b1_v.iter_mut() {
        *x = normal.sample(&mut rand::thread_rng())
    }
    let B1 = Matrix::new(0 as usize, layer_size1 as usize, b1_v);
    println!("{:?}", B1);


    // Train model
//    for i in 0..10000{
//
//
//
//    }

}
