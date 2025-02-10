# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:02:39 2024

@author: Laura
"""

import matplotlib.pyplot as plt


##### EXPERT 1 ######

outliers1_1 = 1-253/1281
outliers1_3 = 1-1396/10438
outliers1_4 = 1-982/20444
outliers1_5 = 1-176/1600

dist1_1_1 = 12.56
dist1_1_3 = 15.41
dist1_1_4 = 15.47
dist1_1_5 = 16.29
dist1_3_3 = 10.62
dist1_3_4 = 11.04
dist1_3_5 = 13.72
dist1_4_4 = 10.97
dist1_4_5 = 13.33
dist1_5_5 = 11.76

##### EXPERT 2 ######

outliers2_1 = 1-446/3253
outliers2_3 = 1-504/6220
outliers2_4 = 1-9/399
outliers2_5 = 1-213/830
dist2_1_1 = 11.42
dist2_1_3 = 12.14
dist2_1_4 = 14.07
dist2_1_5 = 15.08
dist2_3_3 = 11
dist2_3_4 = 13.94
dist2_3_5 = 13.85
dist2_4_4 = 13.55
dist2_4_5 = 15.77
dist2_5_5 = 9.75

##### EXPERT 3 ######

outliers3_1 = 1-1077/5693
outliers3_3 = 1-971/9692
outliers3_4 = 1-707/13627
outliers3_5 = 1-21/43
dist3_1_1 = 11.81
dist3_1_3 = 11.75
dist3_1_4 = 12.71
dist3_1_5 = 15.18
dist3_3_3 = 10.74
dist3_3_4 = 11.82
dist3_3_5 = 14.84
dist3_4_4 = 11.04
dist3_4_5 = 14.34
dist3_5_5 = 9.99

##### EXPERT 4 ######

outliers4_1 = 1-711/5848
outliers4_3 = 1-881/10175
outliers4_4 = 1-826/16808
outliers4_5 = 1-19/155
dist4_1_1 = 13.18
dist4_1_3 = 13.03
dist4_1_4 = 13.54
dist4_1_5 = 16.67
dist4_3_3 = 10.57
dist4_3_4 = 11.40
dist4_3_5 = 15.14
dist4_4_4 = 11.10
dist4_4_5 = 14.45
dist4_5_5 = 12.98

##### EXPERT 5 ######

outliers5_1 = 1-734/4040
outliers5_3 = 1-1129/14440
outliers5_4 = 1-894/17672
outliers5_5 = 1-160/537
dist5_1_1 = 11.85
dist5_1_3 = 13.51
dist5_1_4 = 14.51
dist5_1_5 = 16.61
dist5_3_3 = 10.55
dist5_3_4 = 11.53
dist5_3_5 = 14.06
dist5_4_4 = 10.94
dist5_4_5 = 13.26
dist5_5_5 = 10.15

##### EXPERT 6 ######

outliers6_1 = 1-594/1680
outliers6_3 = 1-216/3189
outliers6_4 = 1-199/4564
outliers6_5 = 1-444/871
dist6_1_1 = 11.40
dist6_1_3 = 13.60
dist6_1_4 = 13.77
dist6_1_5 = 17.08
dist6_3_3 = 11.51
dist6_3_4 = 12.14
dist6_3_5 = 15.89
dist6_4_4 = 11.98
dist6_4_5 = 15.20
dist6_5_5 = 8.92

##### CONSENSUS ######
outlierscons_1 = 1-249/1032
outlierscons_3 = 1-322/1726
outlierscons_4 = 1-369/3401
distcons_1_1 = 12.05
distcons_1_3 = 15.27
distcons_1_4 = 15.47
distcons_3_3 = 10.94
distcons_3_4 = 13.58
distcons_4_4 = 11.45

##### PANDA ######
outliersPANDA_or_1 = 1-910/7063
outliersPANDA_or_2 = 1-43/792
outliersPANDA_or_3 = 1-32/363
outliersPANDA_or_4 = 1-222/1309
outliersPANDA_or_5 = 1-15/42
distPANDA_or_1_1 = 14.33
distPANDA_or_1_2 = 19.13
distPANDA_or_1_3 = 18.98
distPANDA_or_1_4 = 20.19
distPANDA_or_1_5 = 18.58
distPANDA_or_2_2 = 16.68
distPANDA_or_2_3 = 18.62
distPANDA_or_2_4 = 18.88
distPANDA_or_2_5 = 19.09
distPANDA_or_3_3 = 15.51
distPANDA_or_3_4 = 17.55
distPANDA_or_3_5 = 18.04
distPANDA_or_4_4 = 12.63
distPANDA_or_4_5 = 18.18
distPANDA_or_5_5 = 13.35

##### PANDA 1 ######
outliersPANDA_1_1 = 1-910/7063
outliersPANDA_1_2 = 1-35/663
outliersPANDA_1_3 = 1-60/492
outliersPANDA_1_4 = 1-222/1309
outliersPANDA_1_5 = 1-15/42
distPANDA_1_1_1 = 14.33
distPANDA_1_1_2 = 18.99
distPANDA_1_1_3 = 19.04
distPANDA_1_1_4 = 20.19
distPANDA_1_1_5 = 18.58
distPANDA_1_2_2 = 16.74
distPANDA_1_2_3 = 18.26
distPANDA_1_2_4 = 17.70
distPANDA_1_2_5 = 18.05
distPANDA_1_3_3 = 16.14
distPANDA_1_3_4 = 18.70
distPANDA_1_3_5 = 18.68
distPANDA_1_4_4 = 12.63
distPANDA_1_4_5 = 18.18
distPANDA_1_5_5 = 13.35

##### PANDA 2 ######
outliersPANDA_2_1 = 1-910/7063
outliersPANDA_2_2 = 1-43/792
outliersPANDA_2_3 = 1-327/647
outliersPANDA_2_4 = 1-133/1025
outliersPANDA_2_5 = 1-15/42
distPANDA_2_1_1 = 14.33
distPANDA_2_1_2 = 19.13
distPANDA_2_1_3 = 20.34
distPANDA_2_1_4 = 18.93
distPANDA_2_1_5 = 18.58
distPANDA_2_2_2 = 16.68
distPANDA_2_2_3 = 18.88
distPANDA_2_2_4 = 18.70
distPANDA_2_2_5 = 19.09
distPANDA_2_3_3 = 11.31
distPANDA_2_3_4 = 17.70
distPANDA_2_3_5 = 18.04
distPANDA_2_4_4 = 15.75
distPANDA_2_4_5 = 18.48
distPANDA_2_5_5 = 13.35

##### PANDA 3 ######
outliersPANDA_3_1 = 1-910/7063
outliersPANDA_3_2 = 1-13/706
outliersPANDA_3_3 = 1-282/637
outliersPANDA_3_4 = 1-115/1121
outliersPANDA_3_5 = 1-15/42
distPANDA_3_1_1 = 14.33
distPANDA_3_1_2 = 18.73
distPANDA_3_1_3 = 20.19
distPANDA_3_1_4 = 19.09
distPANDA_3_1_5 = 18.58
distPANDA_3_2_2 = 16.51
distPANDA_3_2_3 = 18.55
distPANDA_3_2_4 = 18.29
distPANDA_3_2_5 = 19.36
distPANDA_3_3_3 = 12.91
distPANDA_3_3_4 = 17.39
distPANDA_3_3_5 = 18.23
distPANDA_3_4_4 = 15.82
distPANDA_3_4_5 = 18.56
distPANDA_3_5_5 = 13.35


def compute_sc(means, outliers_sc):
    dist_sc = min(means[1:]) - means[0] 
    sc = outliers_sc * dist_sc
    return sc,dist_sc

def compute_min_dist(means,means_cl):
    dists = []
    for i,mean in enumerate(means):
        dists.append(2*mean-means_cl[0]-means_cl[i+1])
    return min(dists),dists

dist1_1,dists1_1 = compute_min_dist([dist1_1_3,dist1_1_4,dist1_1_5],[dist1_1_1,dist1_3_3,dist1_4_4,dist1_5_5])
dist1_3,dists1_3 = compute_min_dist([dist1_1_3,dist1_3_4,dist1_3_5],[dist1_3_3,dist1_1_1,dist1_4_4,dist1_5_5])
dist1_4,dists1_4 = compute_min_dist([dist1_1_4,dist1_3_4,dist1_4_5],[dist1_4_4,dist1_1_1,dist1_3_3,dist1_5_5])
dist1_5,dists1_5 = compute_min_dist([dist1_1_5,dist1_3_5,dist1_4_5],[dist1_5_5,dist1_1_1,dist1_3_3,dist1_4_4])

dist2_1 = compute_min_dist([dist2_1_3,dist2_1_4,dist2_1_5],[dist2_1_1,dist2_3_3,dist2_4_4,dist2_5_5])
dist2_3 = compute_min_dist([dist2_1_3,dist2_3_4,dist2_3_5],[dist2_3_3,dist2_1_1,dist2_4_4,dist2_5_5])
dist2_4 = compute_min_dist([dist2_1_4,dist2_3_4,dist2_4_5],[dist2_4_4,dist2_1_1,dist2_3_3,dist2_5_5])
dist2_5 = compute_min_dist([dist2_1_5,dist2_3_5,dist2_4_5],[dist2_5_5,dist2_1_1,dist2_3_3,dist2_4_4])

dist3_1 = compute_min_dist([dist3_1_3,dist3_1_4,dist3_1_5],[dist3_1_1,dist3_3_3,dist3_4_4,dist3_5_5])
dist3_3 = compute_min_dist([dist3_1_3,dist3_3_4,dist3_3_5],[dist3_3_3,dist3_1_1,dist3_4_4,dist3_5_5])
dist3_4 = compute_min_dist([dist3_1_4,dist3_3_4,dist3_4_5],[dist3_4_4,dist3_1_1,dist3_3_3,dist3_5_5])
dist3_5 = compute_min_dist([dist3_1_5,dist3_3_5,dist3_4_5],[dist3_5_5,dist3_1_1,dist3_3_3,dist3_4_4])

dist4_1 = compute_min_dist([dist4_1_3,dist4_1_4,dist4_1_5],[dist4_1_1,dist4_3_3,dist4_4_4,dist4_5_5])
dist4_3 = compute_min_dist([dist4_1_3,dist4_3_4,dist4_3_5],[dist4_3_3,dist4_1_1,dist4_4_4,dist4_5_5])
dist4_4 = compute_min_dist([dist4_1_4,dist4_3_4,dist4_4_5],[dist4_4_4,dist4_1_1,dist4_3_3,dist4_5_5])
dist4_5 = compute_min_dist([dist4_1_5,dist4_3_5,dist4_4_5],[dist4_5_5,dist4_1_1,dist4_3_3,dist4_4_4])

dist5_1 = compute_min_dist([dist5_1_3,dist5_1_4,dist5_1_5],[dist5_1_1,dist5_3_3,dist5_4_4,dist5_5_5])
dist5_3 = compute_min_dist([dist5_1_3,dist5_3_4,dist5_3_5],[dist5_3_3,dist5_1_1,dist5_4_4,dist5_5_5])
dist5_4 = compute_min_dist([dist5_1_4,dist5_3_4,dist5_4_5],[dist5_4_4,dist5_1_1,dist5_3_3,dist5_5_5])
dist5_5 = compute_min_dist([dist5_1_5,dist5_3_5,dist5_4_5],[dist5_5_5,dist5_1_1,dist5_3_3,dist5_4_4])
 
dist6_1 = compute_min_dist([dist6_1_3,dist6_1_4,dist6_1_5],[dist6_1_1,dist6_3_3,dist6_4_4,dist6_5_5])
dist6_3 = compute_min_dist([dist6_1_3,dist6_3_4,dist6_3_5],[dist6_3_3,dist6_1_1,dist6_4_4,dist6_5_5])
dist6_4 = compute_min_dist([dist6_1_4,dist6_3_4,dist6_4_5],[dist6_4_4,dist6_1_1,dist6_3_3,dist6_5_5])
dist6_5 = compute_min_dist([dist6_1_5,dist6_3_5,dist6_4_5],[dist6_5_5,dist6_1_1,dist6_3_3,dist6_4_4])

distcons_1 = compute_min_dist([distcons_1_3,distcons_1_4],[distcons_1_1,distcons_3_3,distcons_4_4])
distcons_3 = compute_min_dist([distcons_1_3,distcons_3_4],[distcons_3_3,distcons_1_1,distcons_4_4])
distcons_4 = compute_min_dist([distcons_1_4,distcons_3_4],[distcons_4_4,distcons_1_1,distcons_3_3])

sc1_1 = outliers1_1 * dist1_1
sc1_3 = outliers1_3 * dist1_3
sc1_4 = outliers1_4 * dist1_4
sc1_5 = outliers1_5 * dist1_5

sc1_1_simple = compute_sc([dist1_1_1,dist1_1_3,dist1_1_4,dist1_1_5], outliers1_1)
sc1_3_simple = compute_sc([dist1_3_3,dist1_1_3,dist1_3_4,dist1_3_5], outliers1_3)
sc1_4_simple = compute_sc([dist1_4_4,dist1_1_4,dist1_3_4,dist1_4_5], outliers1_4)
sc1_5_simple = compute_sc([dist1_5_5,dist1_1_5,dist1_3_5,dist1_4_5], outliers1_5)

sc2_1 = outliers2_1 * dist2_1
sc2_3 = outliers2_3 * dist2_3
sc2_4 = outliers2_4 * dist2_4
sc2_5 = outliers2_5 * dist2_5

sc2_1_simple = compute_sc([dist2_1_1,dist2_1_3,dist2_1_4,dist2_1_5], outliers2_1)
sc2_3_simple = compute_sc([dist2_3_3,dist2_1_3,dist2_3_4,dist2_3_5], outliers2_3)
sc2_4_simple = compute_sc([dist2_4_4,dist2_3_4,dist2_1_4,dist2_4_5], outliers2_4)
sc2_5_simple = compute_sc([dist2_5_5,dist2_3_5,dist2_4_5,dist2_1_5], outliers2_5)

sc3_1 = outliers3_1 * dist3_1
sc3_3 = outliers3_3 * dist3_3
sc3_4 = outliers3_4 * dist3_4
sc3_5 = outliers3_5 * dist3_5

sc3_1_simple = compute_sc([dist3_1_1,dist3_1_3,dist3_1_4,dist3_1_5], outliers3_1)
sc3_3_simple = compute_sc([dist3_3_3,dist3_1_3,dist3_3_4,dist3_3_5], outliers3_3)
sc3_4_simple = compute_sc([dist3_4_4,dist3_1_4,dist3_3_4,dist3_4_5], outliers3_4)
sc3_5_simple = compute_sc([dist3_5_5,dist3_1_5,dist3_3_5,dist3_4_5], outliers3_5)

sc4_1 = outliers4_1 * dist4_1
sc4_3 = outliers4_3 * dist4_3
sc4_4 = outliers4_4 * dist4_4
sc4_5 = outliers4_5 * dist4_5

sc4_1_simple = compute_sc([dist4_1_1,dist4_1_3,dist4_1_4,dist4_1_5], outliers4_1)
sc4_3_simple = compute_sc([dist4_3_3,dist4_1_3,dist4_3_4,dist4_3_5], outliers4_3)
sc4_4_simple = compute_sc([dist4_4_4,dist4_3_4,dist4_1_4,dist4_4_5], outliers4_4)
sc4_5_simple = compute_sc([dist4_5_5,dist4_3_5,dist4_4_5,dist4_1_5], outliers4_5)

sc5_1 = outliers5_1 * dist5_1
sc5_3 = outliers5_3 * dist5_3
sc5_4 = outliers5_4 * dist5_4
sc5_5 = outliers5_5 * dist5_5

sc5_1_simple = compute_sc([dist5_1_1,dist5_1_3,dist5_1_4,dist5_1_5], outliers5_1)
sc5_3_simple = compute_sc([dist5_3_3,dist5_1_3,dist5_3_4,dist5_3_5], outliers5_3)
sc5_4_simple = compute_sc([dist5_4_4,dist5_3_4,dist5_1_4,dist5_4_5], outliers5_4)
sc5_5_simple = compute_sc([dist5_5_5,dist5_3_5,dist5_4_5,dist5_1_5], outliers5_5)

sc6_1 = outliers6_1 * dist6_1
sc6_3 = outliers6_3 * dist6_3
sc6_4 = outliers6_4 * dist6_4
sc6_5 = outliers6_5 * dist6_5

sc6_1_simple = compute_sc([dist6_1_1,dist6_1_3,dist6_1_4,dist6_1_5], outliers6_1)
sc6_3_simple = compute_sc([dist6_3_3,dist6_1_3,dist6_3_4,dist6_3_5], outliers6_3)
sc6_4_simple = compute_sc([dist6_4_4,dist6_3_4,dist6_1_4,dist6_4_5], outliers6_4)
sc6_5_simple = compute_sc([dist6_5_5,dist6_3_5,dist6_4_5,dist6_1_5], outliers6_5)

sccons_1 = outlierscons_1 * distcons_1
sccons_3 = outlierscons_3 * distcons_3
sccons_4 = outlierscons_4 * distcons_4


distPANDA_or_1,distsPANDA_or_1 = compute_min_dist([distPANDA_or_1_2,distPANDA_or_1_3,distPANDA_or_1_4,distPANDA_or_1_5],[distPANDA_or_1_1,distPANDA_or_2_2,distPANDA_or_3_3,distPANDA_or_4_4,distPANDA_or_5_5])
distPANDA_or_2,distsPANDA_or_2 = compute_min_dist([distPANDA_or_1_2,distPANDA_or_2_3,distPANDA_or_2_4,distPANDA_or_2_5],[distPANDA_or_2_2,distPANDA_or_1_1,distPANDA_or_3_3,distPANDA_or_4_4,distPANDA_or_5_5])
distPANDA_or_3,distsPANDA_or_3 = compute_min_dist([distPANDA_or_1_3,distPANDA_or_2_3,distPANDA_or_3_4,distPANDA_or_3_5],[distPANDA_or_3_3,distPANDA_or_1_1,distPANDA_or_2_2,distPANDA_or_4_4,distPANDA_or_5_5])
distPANDA_or_4,distsPANDA_or_4 = compute_min_dist([distPANDA_or_1_4,distPANDA_or_2_4,distPANDA_or_3_4,distPANDA_or_4_5],[distPANDA_or_4_4,distPANDA_or_1_1,distPANDA_or_2_2,distPANDA_or_3_3,distPANDA_or_5_5])
distPANDA_or_5,distsPANDA_or_5 = compute_min_dist([distPANDA_or_1_5,distPANDA_or_2_5,distPANDA_or_3_5,distPANDA_or_4_5],[distPANDA_or_5_5,distPANDA_or_1_1,distPANDA_or_2_2,distPANDA_or_3_3,distPANDA_or_4_4])

distPANDA_1_1,distsPANDA_1_1 = compute_min_dist([distPANDA_1_1_2,distPANDA_1_1_3,distPANDA_1_1_4,distPANDA_1_1_5],[distPANDA_1_1_1,distPANDA_1_2_2,distPANDA_1_3_3,distPANDA_1_4_4,distPANDA_1_5_5])
distPANDA_1_2,distsPANDA_1_2 = compute_min_dist([distPANDA_1_1_2,distPANDA_1_2_3,distPANDA_1_2_4,distPANDA_1_2_5],[distPANDA_1_2_2,distPANDA_1_1_1,distPANDA_1_3_3,distPANDA_1_4_4,distPANDA_1_5_5])
distPANDA_1_3,distsPANDA_1_3 = compute_min_dist([distPANDA_1_1_3,distPANDA_1_2_3,distPANDA_1_3_4,distPANDA_1_3_5],[distPANDA_1_3_3,distPANDA_1_1_1,distPANDA_1_2_2,distPANDA_1_4_4,distPANDA_1_5_5])
distPANDA_1_4,distsPANDA_1_4 = compute_min_dist([distPANDA_1_1_4,distPANDA_1_2_4,distPANDA_1_3_4,distPANDA_1_4_5],[distPANDA_1_4_4,distPANDA_1_1_1,distPANDA_1_2_2,distPANDA_1_3_3,distPANDA_1_5_5])
distPANDA_1_5,distsPANDA_1_5 = compute_min_dist([distPANDA_1_1_5,distPANDA_1_2_5,distPANDA_1_3_5,distPANDA_1_4_5],[distPANDA_1_5_5,distPANDA_1_1_1,distPANDA_1_2_2,distPANDA_1_3_3,distPANDA_1_4_4])

distPANDA_2_1,distsPANDA_2_1 = compute_min_dist([distPANDA_2_1_2,distPANDA_2_1_3,distPANDA_2_1_4,distPANDA_2_1_5],[distPANDA_2_1_1,distPANDA_2_2_2,distPANDA_2_3_3,distPANDA_2_4_4,distPANDA_2_5_5])
distPANDA_2_2,distsPANDA_2_2 = compute_min_dist([distPANDA_2_1_2,distPANDA_2_2_3,distPANDA_2_2_4,distPANDA_2_2_5],[distPANDA_2_2_2,distPANDA_2_1_1,distPANDA_2_3_3,distPANDA_2_4_4,distPANDA_2_5_5])
distPANDA_2_3,distsPANDA_2_3 = compute_min_dist([distPANDA_2_1_3,distPANDA_2_2_3,distPANDA_2_3_4,distPANDA_2_3_5],[distPANDA_2_3_3,distPANDA_2_1_1,distPANDA_2_2_2,distPANDA_2_4_4,distPANDA_2_5_5])
distPANDA_2_4,distsPANDA_2_4 = compute_min_dist([distPANDA_2_1_4,distPANDA_2_2_4,distPANDA_2_3_4,distPANDA_2_4_5],[distPANDA_2_4_4,distPANDA_2_1_1,distPANDA_2_2_2,distPANDA_2_3_3,distPANDA_2_5_5])
distPANDA_2_5,distsPANDA_2_5 = compute_min_dist([distPANDA_2_1_5,distPANDA_2_2_5,distPANDA_2_3_5,distPANDA_2_4_5],[distPANDA_2_5_5,distPANDA_2_1_1,distPANDA_2_2_2,distPANDA_2_3_3,distPANDA_2_4_4])

distPANDA_3_1,distsPANDA_3_1 = compute_min_dist([distPANDA_3_1_2,distPANDA_3_1_3,distPANDA_3_1_4,distPANDA_3_1_5],[distPANDA_3_1_1,distPANDA_3_2_2,distPANDA_3_3_3,distPANDA_3_4_4,distPANDA_3_5_5])
distPANDA_3_2,distsPANDA_3_2 = compute_min_dist([distPANDA_3_1_2,distPANDA_3_2_3,distPANDA_3_2_4,distPANDA_3_2_5],[distPANDA_3_2_2,distPANDA_3_1_1,distPANDA_3_3_3,distPANDA_3_4_4,distPANDA_3_5_5])
distPANDA_3_3,distsPANDA_3_3 = compute_min_dist([distPANDA_3_1_3,distPANDA_3_2_3,distPANDA_3_3_4,distPANDA_3_3_5],[distPANDA_3_3_3,distPANDA_3_1_1,distPANDA_3_2_2,distPANDA_3_4_4,distPANDA_3_5_5])
distPANDA_3_4,distsPANDA_3_4 = compute_min_dist([distPANDA_3_1_4,distPANDA_3_2_4,distPANDA_3_3_4,distPANDA_3_4_5],[distPANDA_3_4_4,distPANDA_3_1_1,distPANDA_3_2_2,distPANDA_3_3_3,distPANDA_3_5_5])
distPANDA_3_5,distsPANDA_3_5 = compute_min_dist([distPANDA_3_1_5,distPANDA_3_2_5,distPANDA_3_3_5,distPANDA_3_4_5],[distPANDA_3_5_5,distPANDA_3_1_1,distPANDA_3_2_2,distPANDA_3_3_3,distPANDA_3_4_4])


scPANDA_or_1 = outliersPANDA_or_1 * distPANDA_or_1
scPANDA_or_2 = outliersPANDA_or_2 * distPANDA_or_2
scPANDA_or_3 = outliersPANDA_or_3 * distPANDA_or_3
scPANDA_or_4 = outliersPANDA_or_4 * distPANDA_or_4
scPANDA_or_5 = outliersPANDA_or_5 * distPANDA_or_5

scPANDA_1_1 = outliersPANDA_1_1 * distPANDA_1_1
scPANDA_1_2 = outliersPANDA_1_2 * distPANDA_1_2
scPANDA_1_3 = outliersPANDA_1_3 * distPANDA_1_3
scPANDA_1_4 = outliersPANDA_1_4 * distPANDA_1_4
scPANDA_1_5 = outliersPANDA_1_5 * distPANDA_1_5

scPANDA_2_1 = outliersPANDA_1_2 * distPANDA_2_1
scPANDA_2_2 = outliersPANDA_2_2 * distPANDA_2_2
scPANDA_2_3 = outliersPANDA_2_3 * distPANDA_2_3
scPANDA_2_4 = outliersPANDA_2_4 * distPANDA_2_4
scPANDA_2_5 = outliersPANDA_2_5 * distPANDA_2_5

scPANDA_3_1 = outliersPANDA_3_1 * distPANDA_3_1
scPANDA_3_2 = outliersPANDA_3_2 * distPANDA_3_2
scPANDA_3_3 = outliersPANDA_3_3 * distPANDA_3_3
scPANDA_3_4 = outliersPANDA_3_4 * distPANDA_3_4
scPANDA_3_5 = outliersPANDA_3_5 * distPANDA_3_5



outliers1 = [outliers1_1,outliers1_3,outliers1_4,outliers1_5]
outliers2 = [outliers2_1,outliers2_3,outliers2_4,outliers2_5]
outliers3 = [outliers3_1,outliers3_3,outliers3_4,outliers3_5]
outliers4 = [outliers4_1,outliers4_3,outliers4_4,outliers4_5]
outliers5 = [outliers5_1,outliers5_3,outliers5_4,outliers5_5]
outliers6 = [outliers6_1,outliers6_3,outliers6_4,outliers6_5]

distances1 = [dist1_1,dist1_3,dist1_4,dist1_5]
distances2 = [dist2_1,dist2_3,dist2_4,dist2_5]
distances3 = [dist3_1,dist3_3,dist3_4,dist3_5]
distances4 = [dist4_1,dist4_3,dist4_4,dist4_5]
distances5 = [dist5_1,dist5_3,dist5_4,dist5_5]
distances6 = [dist6_1,dist6_3,dist6_4,dist6_5]

plt.scatter(outliers1, distances1,c='yellow',marker='o',label='expert 1')
plt.scatter(outliers2, distances2,c='blue',marker='v',label='expert 2')
plt.scatter(outliers3, distances3,c='green',marker = 's',label='expert 3')
plt.scatter(outliers4, distances4,c='red',marker='*',label='expert 4')
plt.scatter(outliers5, distances5,c='pink',marker='x',label='expert 5')
plt.scatter(outliers6, distances6,c='orange',marker = '+',label='expert 6')
plt.legend()
plt.xlabel("1-(N_o/N_t)")
plt.ylabel("min dist")