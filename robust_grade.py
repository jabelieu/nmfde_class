import numpy as np
import pandas as pd

resolution = 5 + 1
final_score_threshold = 0.8

hw1_w = .1
hw2_w = .1
hw3_w = .1
hw4_w = .1

midterm_exam_w = .3
final_exam_w = .2
final_proj_w = .3

weights = [  hw1_w , hw2_w , hw3_w , hw4_w ,
            midterm_exam_w ,final_proj_w ]

# hw1_score = np.linspace ( 0. , 1. , resolution )
hw1_score = .96 * np.ones ( resolution )
hw2_score = np.linspace ( 0. , 1. , resolution )
hw3_score = np.linspace ( 0. , 0.95 , resolution )
hw4_score = hw1_score.copy()

# midterm_exam_score = np.linspace ( 0. , 1. , resolution )
midterm_exam_score =.70 * np.ones ( resolution )
final_exam_score = np.linspace ( 0. , 1. , resolution )
final_proj_score = np.linspace ( 0. , 1.0 , resolution )

hw1s, hw2s, hw3s, hw4s, midterms, finalprojs = np.meshgrid (
    hw1_score , hw2_score , hw3_score , hw4_score ,
    midterm_exam_score , final_proj_score ,
    indexing='ij' )

all_scores = np.array ( [ hw1s , hw2s , hw3s , hw4s ,
                          midterms , finalprojs ] )

total_score = np.tensordot(weights, all_scores, axes=1)

mask = total_score >= final_score_threshold

valid_hw1  = hw1s[mask]
valid_hw2  = hw2s[mask]
valid_hw3  = hw3s[mask]
valid_hw4  = hw4s[mask]
valid_mid  = midterms[mask]
# valid_final = finals[mask]
valid_proj = finalprojs[mask]
valid_total = total_score[mask]

df = pd.DataFrame({
    'hw1': valid_hw1,
    'hw2': valid_hw2,
    'hw3': valid_hw3,
    'hw4': valid_hw4,
    'midterm': valid_mid,
    # 'final_exam': valid_final,
    'final_project': valid_proj,
    'total_score' : valid_total
})
df = df.sort_values(by='total_score',ascending=True)
print(df)