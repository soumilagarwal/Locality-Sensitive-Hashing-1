import time
import shingler
import minhashing
import lsh
import numpy as np
import pandas as pd


def main():
    data = pd.read_csv('human_data.csv')
    sequences=data['sequence']
    # data = pd.read_csv("test_data.txt")
    # sequences = data['document']
    print("Number of documents = ", len(sequences))

    start = time.time()
    shingles=[]
    shingle_matrix = []
    shingle_size = 4
    shingles = shingler.generate_shingles(sequences, shingle_size)
    shingle_matrix = shingler.initialize_matrix(sequences, shingles)    #shingle matrix 336x4380
    end = time.time()

    print("Number of shingles found = ", len(shingles))
    # print(shingles)
    print("Shingle matrix size is ")
    print(len(shingles), " x ", len(shingle_matrix[0]))
    print("Time taken for shingling = ",round((end - start),2),"s\n")

    start = time.time()
    sig_matrix = minhashing.minhasher(shingles, shingle_matrix)
    end = time.time()

    print("Signature matrix size is ")
    print(len(sig_matrix), " x ", len(sig_matrix[0]))
    # print("signature matrix is\n",sig_matrix)
    print("Time taken for minhashing = ",round((end - start),2),"s\n")

    start = time.time()
    buckets_list = lsh.bucket_list(sig_matrix)
    end = time.time()
    print("Buckets generated successfully.")
    # print("Bucket lists are \n",buckets_list)
    print("Time taken for bucket generation = ",round((end - start),2),"s\n")
    
    #query_dna = input("Enter DNA sequence : ") 
    #sim_docs = lsh.find_sim_docs(query_dna, buckets_list, sig_matrix)
    # final = similarity.compute_similarity(query_dna, sim_docs, shingle_matrix)
if __name__ == "__main__":
    main()
