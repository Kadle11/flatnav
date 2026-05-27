# Convert one bvecs file to fvecs using the suffix from the input name.
echo "Converting to float32 fvecs format..."
python3 <<EOF
import numpy as np
import os

def convert_bvecs_to_fvecs(input_file, output_file):
    print(f"Converting {input_file} to {output_file}...")
    # Each vector is 1 byte header (int) + 128 uint8
    # We use memmap to handle large files without crashing small RAM systems
    dim = 128
    vector_size = 4 + dim # 4 bytes for dim int + 128 bytes data
    
    # Calculate number of vectors
    file_size = os.path.getsize(input_file)
    num_vecs = file_size // vector_size
    
    # Read as uint8, but skip the 4-byte 'dim' headers during conversion
    data_raw = np.memmap(input_file, dtype='uint8', mode='r', shape=(num_vecs, vector_size))
    
    # Slice out the 4-byte headers and convert the remaining 128 dims to float32
    # Then write to fvecs format: [4-byte dim][128 * 4-byte floats]
    with open(output_file, 'wb') as f:
        for i in range(num_vecs):
            # Write 4-byte header for float32 (still 128)
            f.write(np.int32(dim).tobytes())
            # Convert and write data
            vec_data = data_raw[i, 4:].astype('float32')
            f.write(vec_data.tobytes())
            if i % 10000000 == 0:
                print(f"Processed {i} vectors...")

def convert_dataset(input_suffix):
    dataset_dir = f'sift-128-euclidean-{input_suffix}'
    convert_bvecs_to_fvecs(
        f'{dataset_dir}/sift{input_suffix}_base.bvecs',
        f'{dataset_dir}/sift{input_suffix}_base.fvecs',
    )

convert_bvecs_to_fvecs('bigann_query.bvecs', 'sift100m_query.fvecs')
for suffix in ('1M', '5M', '10M', '20M'):
    convert_dataset(suffix)
#convert_bvecs_to_fvecs('extra_queries_200k.bvecs', 'sift100m_200k_extra_query.fvecs')
EOF

echo "Done. Base files written under sift-128-euclidean-1M/, sift-128-euclidean-5M/, sift-128-euclidean-10M/, and sift100m/."
echo "Query file: sift100m_query.fvecs (~5MB)"