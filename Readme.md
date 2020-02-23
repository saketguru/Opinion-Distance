Towards Quantifying the Distance between Opinions
===============================

We share the code of our ICWSM 2020 paper : [Towards Quantifying the Distance between Opinions](https://arxiv.org/abs/2001.09879).


Prerequisites:
--------------
1. First, create a fresh virtual environment and install the requirements.

        conda create -n opinion_distance_env python=3.6
        conda activate opinion_distance_env
        pip install -r requirements.txt
    
2. Get the TagME API gcude-token from [here](https://sobigdata.d4science.org/web/tagme/tagme-help) and add the obtained gcude-token in the file present at `files/tagme_gcude_token.txt`.

3.  Download word2vec embeddings from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) and store the file `GoogleNews-vectors-negative300.bin.gz` in `embedding` folder. 


Example Usage:
--------------

You can run Opinion distance on Seanad Abolition Dataset as follows

    
    python src/run_opinion_measure.py --path dataset/CiviQ_Seanad/ --embedding_strategy word2vec --semantic_threshold 0.6 --baselines False

The above code will generate 
1. Clustering result in folder `dataset/CiviQ_Seanad/results`
2. Compute opinion distance matrices in folder`dataset/CiviQ_Seanad/dist_mats`

Note that the shared code computes opinion distance using OD method mentioned in the paper and uses word2vec embedding strategy. The code requires Internet connection to compute opinion representation using TagME API.

The classification of opinion can be performed by running the below code

    python src/supervised_opinion.py dataset/CiviQ_Seanad/ 

Citing
------
If you find our paper useful in your research, we ask that you cite the following paper:

> Gurukar, Saket, Deepak Ajwani, Sourav Dutta, Juho Lauri, Srinivasan Parthasarathy, and Alessandra Sala. "Towards Quantifying the Distance between Opinions." ICWSM 2020.

    @article{gurukar2020towards,
    title={Towards Quantifying the Distance between Opinions},
    author={Gurukar, Saket and Ajwani, Deepak and Dutta, Sourav and Lauri, Juho and Parthasarathy, Srinivasan and Sala, Alessandra},
    journal={International Conference on Web and Social Media (ICWSM)},
    year={2020}
    }


Contact Us
----------
For questions or comments about the implementation, please contact gurukar.1@osu.edu and deepak.ajwani@ucd.ie.