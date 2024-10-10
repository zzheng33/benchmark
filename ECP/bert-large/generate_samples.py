import tensorflow as tf

def multiply_tfrecords(input_tfrecord, output_tfrecord_base, multipliers):
    # Read the original TFRecord once into memory
    records = []
    for record in tf.data.TFRecordDataset(input_tfrecord):
        records.append(record.numpy())
    
    original_size = len(records)
    print(f"Original number of records: {original_size}")
    
    for multiplier in multipliers:
        output_tfrecord = f"{multiplier * original_size}_samples"
    
        
        with tf.io.TFRecordWriter(output_tfrecord) as writer:
            for _ in range(multiplier):
                for record in records:
                    writer.write(record)

if __name__ == "__main__":
    input_tfrecord = "6000_samples"  # Path to your input TFRecord file
    output_tfrecord_base = "output"  # Base name for the output TFRecord files
    
    # Multipliers to increase the TFRecords (e.g., 2x, 4x, 8x, etc.)
    multipliers = [2, 4, 8, 16, 32]
    
    # Call the function to create multiple output files with multiplied records
    multiply_tfrecords(input_tfrecord, output_tfrecord_base, multipliers)
    
    print("All multiplied TFRecords have been generated.")
