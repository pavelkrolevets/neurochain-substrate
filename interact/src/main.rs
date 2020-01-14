use std::error::Error;
use std::io;
use std::process;

use std::num::ParseIntError;
mod lib;

extern crate mnist;
extern crate rulinalg;
extern crate rand;
extern crate rand_distr;
extern crate hex;

use hex::*;
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




fn main() {

    let x_train = lib::read_mnist().unwrap();
    println!("{:?}", hex::encode(&x_train[0]));


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



}
