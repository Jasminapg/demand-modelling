# SEIR model functions 26/04/2020
# WEL Financial Strategy Team, WEL CCGs (Tower Hamlets, Newham, Waltham Forest CCGs)
# Written by: Nathan Cheetham, Senior Financial Strategy Analyst, nathan.cheetham1@nhs.net
# Adapted from: https://github.com/gabgoh/epcalc/blob/master/src/App.svelte

#%% import packages
import numpy as np
from scipy.integrate import odeint

##%% Fixed model inputs
I0                = 1 # number of initial infections
duration          = 7*12*1e10

#%% Define SEIR model ordinary differential equations (ODEs)
def SEIR_function(x,t,InterventionTime,InterventionTime2,duration,InterventionAmt,InterventionAmt2,R0,D_incubation,D_infectious,
                 P_SEVERE,CFR,D_recovery_mild,D_hospital_lag_severe,D_hospital_lag_fatal,D_recovery_severe,D_death):
    # define when to apply interventions
    if (t > InterventionTime and t <= InterventionTime2):
        beta = (InterventionAmt)*R0/(D_infectious)
    elif (t > InterventionTime2 and t < InterventionTime2 + duration):
        beta = (InterventionAmt2)*R0/(D_infectious)
    elif (t > InterventionTime + duration):
        beta = 0.5*R0/(D_infectious)
    else:
        beta = R0/(D_infectious)
        
    # transmission dynamics
    alpha     = 1/D_incubation # alpha, inverse of incubation time
    gamma = 1/D_infectious # gamma, inverse of infectious time
    p_severe = P_SEVERE # fraction of severe cases
    p_fatal  = CFR # case fatality ratio, fraction of fatal cases
    p_mild   = 1 - P_SEVERE - CFR # fraction of mild cases

    # output populations
    S        = x[0] #// Susceptible
    E        = x[1] #// Exposed
    I        = x[2] #// Infectious 
    Mild     = x[3] #// Recovering (Mild)     
    PreHospital_Severe = x[4] #// Pre-hospital (severe and fatal)
    PreHospital_Fatal = x[5] #// Pre-hospital (severe and fatal)
    Severe_H = x[6] #// Recovering (Severe in hospital)
    Fatal  = x[7] #// Recovering (Fatal in hospital)
    R_Mild   = x[8] #// Recovered (Mild)
    R_Severe = x[9] #// Recovered (Severe) == Discharged from Hospital
    R_Fatal  = x[10] #// Dead
      
    # differential equations for output populations
    dS        = -beta*I*S
    dE        =  beta*I*S - alpha*E
    dI        =  alpha*E - gamma*I
    dMild     =  p_mild*gamma*I   - (1/D_recovery_mild)*Mild
    dPreHospital_Severe   =  p_severe*gamma*I - (1/D_hospital_lag_severe)*PreHospital_Severe
    dPreHospital_Fatal  =  p_fatal*gamma*I - (1/D_hospital_lag_fatal)*PreHospital_Fatal
    dSevere_H =  (1/D_hospital_lag_severe)*PreHospital_Severe - (1/D_recovery_severe)*Severe_H
    dFatal    =  (1/D_hospital_lag_fatal)*PreHospital_Fatal - (1/D_death)*Fatal
    dR_Mild   =  (1/D_recovery_mild)*Mild
    dR_Severe =  (1/D_recovery_severe)*Severe_H
    dR_Fatal  =  (1/D_death)*Fatal    
    
    #        0   1   2   3      4                   5                   6          7       8        9          10
    return [dS, dE, dI, dMild, dPreHospital_Severe, dPreHospital_Fatal, dSevere_H, dFatal, dR_Mild, dR_Severe, dR_Fatal]


#%% Define function to solve differential equations and get S E I R populations
def solve_ode(t,InterventionTime,InterventionTime2,duration,InterventionAmt,InterventionAmt2,R0,D_incubation,D_infectious,
                 P_SEVERE,CFR,D_recovery_mild,D_hospital_lag_severe,D_hospital_lag_fatal,D_recovery_severe,D_death,N):
    # initial conditions
    #        0      1   2    3      4                    5                   6          7        8       9          10
    # x    [dS,    dE, dI,   dMild, dPreHospital_Severe, dPreHospital_Fatal, dSevere_H, dFatal, dR_Mild, dR_Severe, dR_Fatal]
    x0 =   [1-I0/N, 0, I0/N, 0,     0,                   0,                  0,         0,      0,       0,         0]
    # solve ode
    x = N*odeint(SEIR_function,x0,t,args = (InterventionTime,InterventionTime2,duration,InterventionAmt,InterventionAmt2,R0,D_incubation,D_infectious,
                 P_SEVERE,CFR,D_recovery_mild,D_hospital_lag_severe,D_hospital_lag_fatal,D_recovery_severe,D_death))
    return x

#%% Define function that solves differential equations and gets S E I R populations, then adds other relevant columns 

def SEIR_results(t,N,ICU_prop,
                 InterventionTime,InterventionTime2,InterventionAmt,InterventionAmt2,R0,D_incubation,D_infectious,
                 P_SEVERE,CFR,D_recovery_mild,D_hospital_lag_severe,D_hospital_lag_fatal,D_recovery_severe,D_death):
      
    # set initial conditions
    #        0      1   2    3      4                    5                   6          7        8       9          10
    # x    [dS,    dE, dI,   dMild, dPreHospital_Severe, dPreHospital_Fatal, dSevere_H, dFatal, dR_Mild, dR_Severe, dR_Fatal]
    x0 =   [1-I0/N, 0, I0/N, 0,     0,                   0,                  0,         0,      0,       0,         0]
    
    # solve differential equations
    x = N*odeint(SEIR_function,x0,t,args = (InterventionTime,InterventionTime2,duration,InterventionAmt,InterventionAmt2,R0,D_incubation,D_infectious,
                 P_SEVERE,CFR,D_recovery_mild,D_hospital_lag_severe,D_hospital_lag_fatal,D_recovery_severe,D_death))
        
    # define additional columns
    recovered_all_col = np.array(x[:,8] + x[:,9]) # recovered (mild) + recovered (severe (hospitalised))
    hospitalised_col = np.array(x[:,6] + x[:,7]) # severe (hospitalised) + fatal
    ICU_col = hospitalised_col * ICU_prop # hospitalised column * proportion needing ICU
    
    # Below columns need to be re-calculated after interpolation, in order to get daily changes
    discharge_daily_col = np.diff(x[:,9]) # rate of change in discharged patients, based on stepwise increase in recovered from hospital
    discharge_daily_col = np.insert(discharge_daily_col,0,0) # insert 0 as first value in daily column
    fatality_daily_col = np.diff(x[:,10]) # rate of change in fatalities, from stepwise increase in deaths
    fatality_daily_col = np.insert(fatality_daily_col,0,0) # insert 0 as first value in daily column
    
    # column to be equivalent of PHE data (assuming only hospitalised cases confirmed) 
    # PHE = Hospitalised (severe+fatal) + Cumulative Recovered (from severe hospitalised) + Cumulative Deaths (from severe)
    # This assumption is becoming increasingly invalid as the testing becomes wider, covering more patients than solely those who are hospitalised. 
    PHE_equivalent_column = hospitalised_col + x[:,9] + x[:,10]
    
    # append new columns to table
    x = np.column_stack((x, recovered_all_col)) # col 11
    x = np.column_stack((x, hospitalised_col)) # col 12
    x = np.column_stack((x, ICU_col)) # col 13
    x = np.column_stack((x, discharge_daily_col)) # col 14
    x = np.column_stack((x, fatality_daily_col)) # col 15
    x = np.column_stack((x, PHE_equivalent_column)) # col 16
    
    return x


#%% Define function that solves SEIR differential equations for multiple sub-populations and adds together
# NOT CURRENTLY USED. 
def SEIR_subpop_sum(t,col_names,data_pop,data_pop_age,data_hosp_age,region_idx,
                    InterventionTime,InterventionTime2,InterventionAmt,InterventionAmt2,R0,D_incubation,D_infectious,
                    P_SEVERE,CFR,D_recovery_mild,D_hospital_lag_severe,D_hospital_lag_fatal,D_recovery_severe,D_death):
    SEIR_subtable = np.zeros([t.shape[0],len(col_names),data_pop_age.shape[0]]) # create empty tables/lists to fill with final data
               
    for s in range(0,data_pop_age.shape[0]-1,1): # for each sub-population age group, calculate individual hospitalised column
        N = (data_pop_age.iloc[s,region_idx+1]/data_pop_age.iloc[-1,region_idx+1]) * data_pop.iloc[0,region_idx] # subpopulation size of age group
        P_SEVERE_subpop = data_hosp_age.iloc[s,1]
        ICU_prop = data_hosp_age.iloc[s,2]
        CFR_subpop = data_hosp_age.iloc[s,3]
        
        x = SEIR_results(t,N,ICU_prop,
                 InterventionTime,InterventionTime2,InterventionAmt,InterventionAmt2,R0,D_incubation,D_infectious,
                 P_SEVERE_subpop,CFR_subpop,D_recovery_mild,D_hospital_lag_severe,D_hospital_lag_fatal,D_recovery_severe,D_death)
        
        SEIR_subtable[:,:,s] = x
        
    # add together columns for various age groups
    SEIR_table = np.sum(SEIR_subtable[:,:,0:data_pop_age.shape[0]-1], axis = 2)

    return SEIR_table
