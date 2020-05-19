# SEIR model functions 12/05/2020
# WEL Financial Strategy Team, WEL CCGs (Tower Hamlets, Newham, Waltham Forest CCGs)
# Written by: Nathan Cheetham, Senior Financial Strategy Analyst, nathan.cheetham1@nhs.net
# Adapted from: https://github.com/gabgoh/epcalc/blob/master/src/App.svelte

#%% import packages
import numpy as np
from scipy.integrate import odeint

##%% Fixed model inputs
I0                = 1 # number of initial infections
duration          = 7*12*1e10

idx_hospitalised_GA = 7
idx_hospitalised_ICU = 8
idx_R_asym = 9
idx_R_mild = 10
idx_R_hospitalised = 11
idx_R_fatal = 12

hosp_col_idx =  14 # column index of all hospitalised column
discharge_daily_col_idx = 15 # idx of daily discharges col
fatality_daily_col_idx = 16 # idx of daily deaths col
idx_cases = 17

# Column headings 
col_names = ['Susceptible','Exposed','Infectious (symptomatic)','Infectious (symptomatic)','Asymptomatic','Mild','Pre-hospital','Hospitalised (GA)','Hospitalised (ICU)'
                 ,'Recovered (asymptomatic)','Recovered (mild)','Discharged (cumulative)','Deaths (cumulative)'
                 , 'Recovered (all)', 'Hospitalised (all)', 'Discharged (daily)'
                 , 'Deaths (daily)', 'Total confirmed cases']

#%% Define SEIR model ordinary differential equations (ODEs)
def SEIR_function(x,t,InterventionTime,InterventionTime2,InterventionTime3,InterventionTime4,duration,InterventionAmt,InterventionAmt2,InterventionAmt3,InterventionAmt4,R0,D_incubation,D_infectious,D_infectious_asym,
                 P_SEVERE,CFR_GA,CFR_ICU,p_ICU,p_sym,D_recovery_asym,D_recovery_mild,D_hospital_lag_GA,D_hospital_lag_ICU,D_recovery_GA,D_recovery_ICU,D_death_GA,D_death_ICU):
    # transmission dynamics
    alpha     = 1/D_incubation # alpha, inverse of incubation time
    gamma = 1/D_infectious # gamma, inverse of infectious time
    gamma_asym = 1/D_infectious_asym # gamma, inverse of infectious time
    p_severe = P_SEVERE # fraction of symptomatic cases requiring hospitalisation
    p_mild   = 1 - P_SEVERE # fraction of symptomatic cases not requiring hospitalisation
    
    p_fatal_GA  = CFR_GA # fraction of G&A hospitalised cases that lead to death
    p_fatal_ICU  = CFR_ICU # fraction of ICU hospitalised cases that lead to death
    
    # p_ICU is fraction of hospitalised cases where patient visits ICU ward
    # p_sym is the fraction of cases that are symptomatic 
    
    # define when to apply interventions
    if (t > InterventionTime and t <= InterventionTime2):
        beta = (InterventionAmt)*R0*(1/((p_sym/gamma) + ((1-p_sym)/gamma_asym)))
    elif (t > InterventionTime2 and t <= InterventionTime3):
        beta = (InterventionAmt2)*R0*(1/((p_sym/gamma) + ((1-p_sym)/gamma_asym)))
    elif (t > InterventionTime3 and t <= InterventionTime4):
        beta = (InterventionAmt3)*R0*(1/((p_sym/gamma) + ((1-p_sym)/gamma_asym)))
    elif (t > InterventionTime4 and t < InterventionTime4 + duration):
        beta = (InterventionAmt4)*R0*(1/((p_sym/gamma) + ((1-p_sym)/gamma_asym)))
    elif (t > InterventionTime + duration):
        beta = 0.5*R0*(1/((p_sym/gamma) + ((1-p_sym)/gamma_asym)))
    else:
        beta = R0*(1/((p_sym/gamma) + ((1-p_sym)/gamma_asym)))
            
    # R0 = p_sym*R0_sym + (1-p_sym)*R0_asym # R0 as the 'effective' R0, based on R0 for symptomatic and asymptomatic cases
    
    
    # output populations
    S        = x[0] #// Susceptible
    E        = x[1] #// Exposed
    I        = x[2] #// Infectious (symptomatic)
    I_asym   = x[3] #// Infectious (asymptomatic)
    Asym     = x[4] #// Recovering (Asymptomatic) 
    Mild     = x[5] #// Recovering (Mild)     
    PreHospital = x[6] #// Pre-hospital
    Severe_GA = x[7] #// Hospital (GA)
    Severe_ICU = x[8] #// Hospital (ICU)
    R_Asym   = x[9] #// Recovered (Asymptomatic)
    R_Mild   = x[10] #// Recovered (Mild)
    R_Severe = x[11] #// Recovered (Severe) == Discharged from Hospital
    R_Fatal  = x[12] #// Died
      
    # differential equations for output populations
    dS        = -beta*I*S -beta*I_asym*S
    dE        =  beta*I*S + beta*I_asym*S - alpha*E
    dI        =  p_sym*alpha*E - gamma*I    
    dI_asym = (1-p_sym)*alpha*E - gamma_asym*I_asym
    
    dAsym = gamma_asym*I_asym -(1/D_recovery_asym)*Asym
    dMild     =  p_mild*gamma*I - (1/D_recovery_mild)*Mild
    dPreHospital   =  p_severe*gamma*I - (1/D_hospital_lag_GA)*(1-p_ICU)*PreHospital  - (1/D_hospital_lag_ICU)*p_ICU*PreHospital  
    dSevere_GA =  (1/D_hospital_lag_GA)*(1-p_ICU)*PreHospital - (1/D_recovery_GA)*(1-p_fatal_GA)*Severe_GA - (1/D_death_GA)*p_fatal_GA*Severe_GA # GA patients
    dSevere_ICU =  (1/D_hospital_lag_ICU)*p_ICU*PreHospital  - (1/D_recovery_ICU)*(1-p_fatal_ICU)*Severe_ICU - (1/D_death_ICU)*p_fatal_ICU*Severe_ICU # ICU patients
    
    dR_Asym = (1/D_recovery_asym)*Asym
    dR_Mild = (1/D_recovery_mild)*Mild
    dR_Severe =  (1/D_recovery_GA)*(1-p_fatal_GA)*Severe_GA + (1/D_recovery_ICU)*(1-p_fatal_ICU)*Severe_ICU
    dR_Fatal  =  (1/D_death_GA)*p_fatal_GA*Severe_GA + (1/D_death_ICU)*p_fatal_ICU*Severe_ICU
    
    #        0   1   2   3       4      5      6             7           8            9        10       11         12
    return [dS, dE, dI, dI_asym, dAsym, dMild, dPreHospital, dSevere_GA, dSevere_ICU, dR_Asym, dR_Mild, dR_Severe, dR_Fatal]


#%% Define function to solve differential equations and get S E I R populations
def solve_ode(t,InterventionTime,InterventionTime2,InterventionTime3,InterventionTime4,duration,InterventionAmt,InterventionAmt2,InterventionAmt3,InterventionAmt4,R0,D_incubation,D_infectious,D_infectious_asym,
                 P_SEVERE,CFR_GA,CFR_ICU,p_ICU,p_sym,D_recovery_asym,D_recovery_mild,D_hospital_lag_GA,D_hospital_lag_ICU,D_recovery_GA,D_recovery_ICU,D_death_GA,D_death_ICU,N):
    # initial conditions
    #        0      1   2    3        4      5      6             7           8            9        10       11         12        
    # x    [dS,     dE, dI,  dI_asym, dAsym, dMild, dPreHospital, dSevere_GA, dSevere_ICU, dR_Asym, dR_Mild, dR_Severe, dR_Fatal]
    x0 =   [1-I0/N, 0, I0/N, 0,       0,     0,     0,            0,          0,           0,       0,       0,         0     ]
    # solve ode
    x = N*odeint(SEIR_function,x0,t,args = (InterventionTime,InterventionTime2,InterventionTime3,InterventionTime4,duration,InterventionAmt,InterventionAmt2,InterventionAmt3,InterventionAmt4,R0,D_incubation,D_infectious,D_infectious_asym,
                 P_SEVERE,CFR_GA,CFR_ICU,p_ICU,p_sym,D_recovery_asym,D_recovery_mild,D_hospital_lag_GA,D_hospital_lag_ICU,D_recovery_GA,D_recovery_ICU,D_death_GA,D_death_ICU))
    return x

#%% Define function that solves differential equations and gets S E I R populations, then adds other relevant columns 

def SEIR_results(t,N,
                 InterventionTime,InterventionTime2,InterventionTime3,InterventionTime4,InterventionAmt,InterventionAmt2,InterventionAmt3,InterventionAmt4,R0,D_incubation,D_infectious,D_infectious_asym,
                 P_SEVERE,CFR_GA,CFR_ICU,p_ICU,p_sym,D_recovery_asym,D_recovery_mild,D_hospital_lag_GA,D_hospital_lag_ICU,D_recovery_GA,D_recovery_ICU,D_death_GA,D_death_ICU):
      
    # set initial conditions
    #        0      1   2    3        4      5      6             7           8            9        10       11         12        
    # x    [dS,     dE, dI,  dI_asym, dAsym, dMild, dPreHospital, dSevere_GA, dSevere_ICU, dR_Asym, dR_Mild, dR_Severe, dR_Fatal]
    x0 =   [1-I0/N, 0, I0/N, 0,       0,     0,     0,            0,          0,           0,       0,       0,         0     ]
    
    # solve differential equations
    x = N*odeint(SEIR_function,x0,t,args = (InterventionTime,InterventionTime2,InterventionTime3,InterventionTime4,duration,InterventionAmt,InterventionAmt2,InterventionAmt3,InterventionAmt4,R0,D_incubation,D_infectious,D_infectious_asym,
                 P_SEVERE,CFR_GA,CFR_ICU,p_ICU,p_sym,D_recovery_asym,D_recovery_mild,D_hospital_lag_GA,D_hospital_lag_ICU,D_recovery_GA,D_recovery_ICU,D_death_GA,D_death_ICU))
        
    # define additional columns
    recovered_all_col = np.array(x[:,idx_R_asym] + x[:,idx_R_mild] + x[:,idx_R_hospitalised]) # recovered (asymptomatic) + recovered (mild) + recovered (severe (hospitalised))
    hospitalised_col = np.array(x[:,idx_hospitalised_GA] + x[:,idx_hospitalised_ICU]) # hospitalised (GA + ICU)
    
    # Below columns need to be re-calculated after interpolation, in order to get daily changes
    discharge_daily_col = np.diff(x[:,idx_R_hospitalised]) # rate of change in discharged patients, based on stepwise increase in recovered from hospital
    discharge_daily_col = np.insert(discharge_daily_col,0,0) # insert 0 as first value in daily column
    fatality_daily_col = np.diff(x[:,idx_R_fatal]) # rate of change in fatalities, from stepwise increase in deaths
    fatality_daily_col = np.insert(fatality_daily_col,0,0) # insert 0 as first value in daily column
    
    # column to be equivalent of PHE data (assuming only hospitalised cases confirmed) 
    # PHE = Hospitalised (severe+fatal) + Cumulative Recovered (from severe hospitalised) + Cumulative Deaths (from severe)
    # This assumption is becoming increasingly invalid as the testing becomes wider, covering more patients than solely those who are hospitalised. 
    PHE_equivalent_column = hospitalised_col + x[:,idx_R_hospitalised] + x[:,idx_R_fatal]
    
    # append new columns to table
    x = np.column_stack((x, recovered_all_col)) # col 13
    x = np.column_stack((x, hospitalised_col)) # col 14
    x = np.column_stack((x, discharge_daily_col)) # col 15
    x = np.column_stack((x, fatality_daily_col)) # col 16
    x = np.column_stack((x, PHE_equivalent_column)) # col 17
    
    return x


#%% Define function that solves SEIR differential equations for multiple sub-populations and adds together
# NOT CURRENTLY USED. NEEDS TO BE UPDATED WITH NEW ODEs IF WANT TO USE
#def SEIR_subpop_sum(t,col_names,data_pop,data_pop_age,data_hosp_age,region_idx,
#                    InterventionTime,InterventionTime2,InterventionTime3,InterventionTime4,InterventionAmt,InterventionAmt2,InterventionAmt3,InterventionAmt4,R0,D_incubation,D_infectious,D_infectious_asym,
#                    P_SEVERE,CFR_GA,CFR_ICU,p_ICU,p_sym,D_recovery_asym,D_recovery_mild,D_hospital_lag_GA,D_hospital_lag_ICU,D_recovery_GA,D_recovery_ICU,D_death_GA,D_death_ICU):
#    SEIR_subtable = np.zeros([t.shape[0],len(col_names),data_pop_age.shape[0]]) # create empty tables/lists to fill with final data
#               
#    for s in range(0,data_pop_age.shape[0]-1,1): # for each sub-population age group, calculate individual hospitalised column
#        N = (data_pop_age.iloc[s,region_idx+1]/data_pop_age.iloc[-1,region_idx+1]) * data_pop.iloc[0,region_idx] # subpopulation size of age group
#        P_SEVERE_subpop = data_hosp_age.iloc[s,1]
#        ICU_prop = data_hosp_age.iloc[s,2]
#        CFR_subpop = data_hosp_age.iloc[s,3]
#        
#        x = SEIR_results(t,N,ICU_prop,
#                 InterventionTime,InterventionTime2,InterventionTime3,InterventionTime4,InterventionAmt,InterventionAmt2,InterventionAmt3,InterventionAmt4,R0,D_incubation,D_infectious,D_infectious_asym,
#                 P_SEVERE_subpop,CFR_subpop,D_recovery_mild,D_hospital_lag_GA,D_hospital_lag_ICU,D_recovery_GA,D_recovery_ICU,D_death_GA,D_death_ICU)
#        
#        SEIR_subtable[:,:,s] = x
#        
#    # add together columns for various age groups
#    SEIR_table = np.sum(SEIR_subtable[:,:,0:data_pop_age.shape[0]-1], axis = 2)
#
#    return SEIR_table
