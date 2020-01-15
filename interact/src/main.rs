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
    let img = &x_train[0][1..];
    let lbl = &x_train[0][0];
    println!("Len {:?}, Label {:?}", &img.len(), &lbl);


    // get Hex

    println!("Vector hex {:?}", hex::encode(img));

    // Random weights
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let mut weights_init: Vec<u8> = vec![0; 784 as usize];
    for x in weights_init.iter_mut() {
        *x = (normal.sample(&mut rand::thread_rng())) as u8
    }

    println!("Weight hex {:?}", hex::encode(weights_init));


    // reshape vector

    let mut mtx1: Vec<Vec<u8>> = vec!(vec!(0; 28); 28);
    let mut counter = 0 as usize;
    for i in 0..27 {
        for j in 0..27{
            counter = (i*j) as usize;
            mtx1[i][j] = img[counter];
        }
    }
    println!("Mtx rows {:?}, cols {:?}", &mtx1.len(), &mtx1[0].len());

    for i in 0..mtx1.len(){
        println!("{:?}", &mtx1[i]);
    }


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
