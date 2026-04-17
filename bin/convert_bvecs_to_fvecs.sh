# 3. Convert to Full Precision (float32)
echo "Converting to float32 fvecs format..."
python3 <<EOF
import numpy as np
import os

def convert_bvecs_to_fvecs(input_file, output_file):
    print(f"Converting {input_file} to {output_file}...")
    # Each vector is 1 byte header (int) + 128 uint8
    # We use memmap to handle the 13GB file without crashing small RAM systems
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
            if i % 10000000 == 0: print(f"Processed {i} vectors...")

#convert_bvecs_to_fvecs('bigann_query.bvecs', 'sift100m_query.fvecs')
#convert_bvecs_to_fvecs('sift100m_base.bvecs', 'sift100m_base.fvecs')
convert_bvecs_to_fvecs('extra_queries_200k.bvecs', 'sift100m_200k_extra_query.fvecs')
EOF

echo "Done. Base file: sift100m_base.fvecs (~48GB)"
echo "Query file: sift100m_query.fvecs (~5MB)"