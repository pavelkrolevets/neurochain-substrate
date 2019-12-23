use hex;
use std::error::Error;
use std::io;
use std::process;

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
    let a: Vec<u8>;
    let b: &[u8] = &[10u8, 20u8, 30u8];
    let weight: Vec<u8> = b.iter().cloned().collect();
    let data: Vec<u8> = [1,1,2].to_vec();

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

    if let Err(err) = read_csv() {
        println!("error running example: {}", err);
        process::exit(1);
    }

}
