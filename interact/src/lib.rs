use hex;
use std::error::Error;
use std::num::ParseIntError;
use std::fs::File;
use std::io;

extern crate rulinalg;
extern crate rand;
extern crate rand_distr;

use rand::distributions::{Normal, Distribution};
use mnist::{Mnist, MnistBuilder};
use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

pub struct ModelParams{
    pub layer_size1: i32,
    pub layer_size2: i32,
    pub classes: i32,
    pub step_size: f64,
}

pub struct Model {
    pub w1: Matrix<f64>,
    pub w2: Matrix<f64>,
    pub w3: Matrix<f64>,
    pub b1: Matrix<f64>,
    pub b2: Matrix<f64>,
    pub b3: Matrix<f64>,
}

impl Model {
    pub fn initialize (params: ModelParams, trn_size: u32, rows: i32, cols: i32) -> Result<(Model), ParseIntError>{
        ////////////////////////////////
        // Initialize weights and biases
        ////////////////////////////////

        // Random vector for weight initialization mean 0, standard deviation 1
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
        let mut w1_v: Vec<f64> = vec![0.0; ((cols * rows) * params.layer_size1) as usize];
        for x in w1_v.iter_mut() {
            *x = normal.sample(&mut rand::thread_rng())
        }
        let W1 = Matrix::new((cols * rows) as usize, params.layer_size1 as usize, w1_v);
        println!("Weights 1 dimentions {}, {}", W1.rows(), W1.cols());

        let mut b1: Vec<f64> = vec![0.0;  params.layer_size1 as usize];
        let B1 = Matrix::new(1 as usize, params.layer_size1 as usize, b1);
        println!("Intercept 1 dimentions {}, {}", B1.rows(), B1.cols());

        let mut w2_v: Vec<f64> = vec![0.0; (params.layer_size1 * params.layer_size2) as usize];
        for x in w2_v.iter_mut() {
            *x = normal.sample(&mut rand::thread_rng())
        }
        let W2 = Matrix::new(params.layer_size1 as usize, params.layer_size2 as usize, w2_v);
        println!("Weights 2 dimentions {}, {}", W2.rows(), W2.cols());

        let mut b2: Vec<f64> = vec![0.0;  params.layer_size2 as usize];
        let B2 = Matrix::new(1 as usize, params.layer_size2 as usize, b2);
        println!("Intercept 2 dimentions {}, {}", B2.rows(), B2.cols());

        let mut w3_v: Vec<f64> = vec![0.0; (params.layer_size2 * params.classes) as usize];
        for x in w3_v.iter_mut() {
            *x = normal.sample(&mut rand::thread_rng())
        }
        let W3 = Matrix::new(params.layer_size2 as usize, params.classes as usize, w3_v);
        println!("Weights 2 dimentions {}, {}", W2.rows(), W2.cols());

        let mut b3: Vec<f64> = vec![0.0;  params.classes as usize];
        let B3 = Matrix::new(1 as usize, params.classes as usize, b3);
        println!("Intercept 2 dimentions {}, {}", B3.rows(), B3.cols());

        let m = Model {
            w1: W1,
            w2: W2,
            w3: W3,
            b1: B1,
            b2: B2,
            b3: B3,
        };

        Ok(m)
    }
}

pub fn read_mnist()->Result<(Vec<Vec<u8>>), ParseIntError>{
    // READ MNIST DATASET TO ARRAY
    let mut X_train: Vec<Vec<u8>> = Vec::new();

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b',')
        .from_reader(io::stdin());
    let mut counter = 0;
    for result in rdr.records() {
        counter += 1;
        let record = result.unwrap();
        // println!("{:?}", record);
        let mut record_as_vec: Vec<u8> = Vec::new();
        for field in record.iter(){
            let mut val: u8 = field.parse().unwrap();
            &record_as_vec.push(val);
        }
        println!("Parsing record {:?}", counter);
        X_train.push(record_as_vec);

        if counter == 1000 {
            break
        }
    }
    println!(" Finished parsing X_train");
    Ok((X_train))
}

pub fn encode_vector<T>(vector: Vec<T>) -> Result<(), ParseIntError>{
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
    Ok(())
}

fn train_mnist ()-> Result<(), ParseIntError>{
    let (trn_size, rows, cols) = (10_000, 28, 28);
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
    let first_image = trn_img[0];
    println!("The first image is a {}.", first_image);

    //Model parameters
    let layer_size1 = 10;
    let layer_size2 = 10;
    let K = 10;

    let mut X_train: Vec<f64> = Vec::new();
    for i in trn_img.iter(){
        X_train.push(*i as f64/ 255f64);
    }

    let mut Y_train: Vec<i64> = Vec::new();
    for i in trn_lbl.iter(){
        Y_train.push(*i as i64);
    }

    println!("{:?}", Y_train.len());

    // Convert the flattened training images vector to a matrix.
    let trn_img = Matrix::new(trn_size as usize, (cols * rows) as usize, X_train);
    println!("Train img dimentions {}, {}", trn_img.rows(), trn_img.cols());

    let trn_lbl = Matrix::new(1 as usize, trn_size  as usize, Y_train);
    println!("Train lbl dimentions {}, {}", trn_lbl.rows(), trn_lbl.cols());

    let params = ModelParams{
        layer_size1: 10,
        layer_size2: 10,
        classes: 10,
        step_size: 0.01,
    };

    let mut model = Model::initialize(params, trn_size, rows, cols).unwrap();
    println!("{:?}", model.w1.rows());

    let mut hidden1: Matrix<f64> = &trn_img * &model.w1;
    println!("Hidden1 dimentions {}, {}", hidden1.rows(), hidden1.cols());

    // apply Relu
    for i in hidden1.row_iter(){
        println!("{:?}", i);

    }


//    // Train model
//    for i in 0..100{
//
//    // Feed forward
//        let hidden1 = &trn_img * &model.w1 + &model.b1;
//
//        let hidden2 = &hidden1 * &model.w2 + &model.b2;
//        let output = &hidden2 * &model.w3 + &model.b3;
//
//
//    }
    Ok(())
}