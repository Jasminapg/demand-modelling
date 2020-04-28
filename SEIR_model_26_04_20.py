# SEIR model 26/04/2020
# WEL Financial Strategy Team, WEL CCGs (Tower Hamlets, Newham, Waltham Forest CCGs)
# Written by: Nathan Cheetham, Senior Financial Strategy Analyst, nathan.cheetham1@nhs.net
# Adapted from: https://github.com/gabgoh/epcalc/blob/master/src/App.svelte

# Script fits SEIR parameters to best replicate observed data on hospitalisation, discharges and deaths
# in North East London (NEL) hospital trusts (Barts Health, BHRUT, Homerton UT)

#%% import packages
import numpy as np
from scipy import optimize as opt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import SEIR_model_functions_26_04_20 as SEIRfun

import copy
from lmfit import Minimizer, Parameters, fit_report, minimize

#%% Load observed and population data
# load confirmed cases data - Source: https://coronavirus.data.gov.uk/#local-authorities
data_cases = pd.read_excel("NEL_cases_new.xlsx") # new form of reporting from Wed 15th April - reporting by day of test
# load ONS population data - Source: https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections/bulletins/subnationalpopulationprojectionsforengland/2018based
data_pop = pd.read_excel("NEL_pop_ons.xlsx") # ONS mid-year 2019
# Load NHSE deaths (NEL only) - Source: https://www.england.nhs.uk/statistics/statistical-work-areas/covid-19-daily-deaths/
data_deaths = pd.read_excel("NEL_deaths.xlsx")
# Load NHSE/I Covid19 dashboard sitrep (NEL only) - Source: NHSE/I Covid-19 Dashboard (not publicly available, requires authorisation) https://analytics.improvement.nhs.uk/#/views/Covid-19Dashboard/Coverpage
data_sitrep_hosp = pd.read_excel("NEL_sitrep_hospitalisation.xlsx") 
data_sitrep_discharge = pd.read_excel("NEL_sitrep_discharge.xlsx") 

# Load fitted parameters
fit_params = pd.read_csv('model_fitted_parameters_2604.csv', delimiter = ',')


#%% Variables to control what code does:
do_fitting = 0 # = 1 to get code to do the fitting. if not, skips to collation of results based on model fitting parameters file

region = 'Homerton' # choose data to fit to. Option: NEL, Barts, BHRUT, Homerton


#%% SEIR Model input parameters
# CURRENTLY FIXED
I0                = 1 # number of initial infections
D_incubation      = 5.1 # incubation time (days)
D_infectious      = 1 # infectious time (days)


# CURRENTLY FITTED
# Reproduction number prior to intervention
R0                = 3.3 # initial R0 value
## Clinical dynamics inputs
# morbidity statistics
CFR               = 0.01 # fraction of cases that are fatal, case fatality ratio
Time_to_death     = 17.8 # time from end of incubation period to death
D_death           = Time_to_death - D_infectious 
# recovery times
D_recovery_mild   = 7  #- D_infectious # recovery time for mild patients
D_recovery_severe = 3 # 3, 10.4 - D_infectious # recovery time for severe patients (hospital length of stay)

# care statistics
P_SEVERE          = 0.044 # fraction of cases that are severe (require hospitalisation)
D_hospital_lag_severe   = 7 # time from symptoms to hospital (for severe but not fatal cases)
D_hospital_lag_fatal    = 5 # time from symptoms to hospital (for fatal cases)
ICU_prop          = 0.3 # Proportion of hospitalised cases requiring ICU/ITU bed

## Intervention inputs
# 1st Intervention
InterventionTime  = 25  # intervention day
OMInterventionAmt = 0.30 # effect of intervention in reducing transmission = fractional reduction in R0
InterventionAmt   = 1 - OMInterventionAmt
# 2nd Intervention 
InterventionTime2  = 32  # intervention day
OMInterventionAmt2 = 0.50 # effect of intervention in reducing transmission = fractional reduction in R0
InterventionAmt2   = 1 - OMInterventionAmt2

#%% Other inputs
# time axis for fitting
Time              = 220.0 # number of days in the model 
dt                = 0.25 # time step (days)
duration          = 7*12*1e10
t = np.linspace(0.0,Time,int((Time+dt)/dt)) # time points

# Column headings 
col_names = ['Susceptible','Exposed','Infectious','Mild','Pre-hospital (Severe)','Pre-hospital (Fatal)','Hospitalised (severe)','Hospitalised (fatal)'
                 ,'Recovered (mild)','Discharged (cumulative)','Deaths (cumulative)'
                 , 'Recovered (all)', 'Hospitalised (all)', 'Hospitalised (ICU)', 'Discharged (daily)'
                 , 'Deaths (daily)', 'Total confirmed cases']


# column indices for relevant columns for plotting
cases_col_idx = 16 # pick which column to fit actual data to - 11 = hospitalised, 15 = PHE equivalent
hosp_col_idx =  12 # column index of hospitalised column
ICU_col_idx = 13 # column index of ICU column
discharge_daily_col_idx = 14 # idx of daily discharges col
discharge_cum_col_idx = 9 # idx of cumulative discharged col
fatality_daily_col_idx = 15 # idx of daily deaths col
fatality_cum_col_idx = 10 # idx of cumulative deaths col

# create time axis to use for all results
time_min = np.ceil(max(fit_params.iloc[1,1:])) # 43880. first day of model as the highest t0 for all regions, rounded up to nearest integer
time_max = np.floor(Time + min(fit_params.iloc[1,1:])) #time_min + Time - 10 # min(fit_params[:,1]) # final day of model as the lowest t0 + number of days, rounded down to nearest integer
dt_final = 1 # time step (days)
t_final = np.linspace(time_min,time_max,int((time_max-time_min+dt_final)/dt_final)) # final time axis, to use for all results

t0_fit_start = 43896 # start date of actual data for fitting
t0_cutoff = data_sitrep_hosp.Date.iloc[-5] # cut off date for actual data for fitting

# Set dates of interventions based on UK measures
InterventionTime1_day = 43907
InterventionTime2_day = 43914
# 43902 = Thurs 12th March - Self-isolation guidelines
# 43906 = Mon 16th March - Social Distancing measures announced
# 43910 = Fri 20th March - Announcement of school and pub closures
# 43913 = Mon 23rd March - UK lockdown announced


#%% Perform fitting of SEIR model parameters to match observed data
if do_fitting == 1: 
    #%% Filter actual data to select desired time range only
    t0_start_idx_PHE = np.where(data_cases.date == t0_fit_start)[0][0]
    t0_cutoff_idx_PHE = np.where(data_cases.date == t0_cutoff+1)[0][0]
    cases_date = data_cases.date[t0_start_idx_PHE:t0_cutoff_idx_PHE]
    
    t0_cutoff_idx_deaths = np.where(data_deaths.Date == t0_cutoff+1)[0][0]
    date_deaths = data_deaths.Date[0:t0_cutoff_idx_deaths]
    
    t0_cutoff_idx_hosp = np.where(data_sitrep_hosp.Date == t0_cutoff+1)[0][0]
    date_hosp = data_sitrep_hosp.Date[0:t0_cutoff_idx_hosp]
    
    t0_cutoff_idx_discharge = np.where(data_sitrep_discharge.Date == t0_cutoff+1)[0][0]
    date_discharge = data_sitrep_discharge.Date[0:t0_cutoff_idx_discharge]
            
    if region == 'NEL':
        population = data_pop.iloc[0,0] #population
        # PHE cases data
        cases = data_cases.iloc[t0_start_idx_PHE:t0_cutoff_idx_PHE,1]
        # NHSE deaths data (NEL only)
        deaths = data_deaths.CumBHRUTBartsHom[:t0_cutoff_idx_deaths] # NEL cumulative deaths                 
        # NHSE/I Covid-19 Dashboard sitrep data (NEL only)
        # hospitalised 
        hospitalised = data_sitrep_hosp.BHRUT_Barts_Hom[0:t0_cutoff_idx_hosp] # NEL hospitalised
        ICU = data_sitrep_hosp.BHRUT_Barts_Hom_ITU[0:t0_cutoff_idx_hosp] # NEL hospitalised (critical care)
        # discharged
        discharged = data_sitrep_discharge.CumBHRUTBartsHom[0:t0_cutoff_idx_discharge] # NEL discharged
    elif region == 'BHRUT':
        population = data_pop.iloc[0,8] #population
        # PHE cases data
        cases = data_cases.iloc[t0_start_idx_PHE:t0_cutoff_idx_PHE,9]
        # NHSE deaths data (NEL only)
        deaths = data_deaths.CumBHRUT[:t0_cutoff_idx_deaths] # NEL cumulative deaths                 
        # NHSE/I Covid-19 Dashboard sitrep data (NEL only)
        # hospitalised 
        hospitalised = data_sitrep_hosp.BHRUT[0:t0_cutoff_idx_hosp] # NEL hospitalised
        ICU = data_sitrep_hosp.BHRUT_ITU[0:t0_cutoff_idx_hosp] # NEL hospitalised (critical care)
        # discharged
        discharged = data_sitrep_discharge.CumBHRUT[0:t0_cutoff_idx_discharge] # NEL discharged
    elif region == 'Barts':
        population = data_pop.iloc[0,9] #population
        # PHE cases data
        cases = data_cases.iloc[t0_start_idx_PHE:t0_cutoff_idx_PHE,10]
        # NHSE deaths data (NEL only)
        deaths = data_deaths.CumBarts[:t0_cutoff_idx_deaths] # NEL cumulative deaths                 
        # NHSE/I Covid-19 Dashboard sitrep data (NEL only)
        # hospitalised 
        hospitalised = data_sitrep_hosp.Barts[0:t0_cutoff_idx_hosp] # NEL hospitalised
        ICU = data_sitrep_hosp.Barts_ITU[0:t0_cutoff_idx_hosp] # NEL hospitalised (critical care)
        # discharged
        discharged = data_sitrep_discharge.CumBarts[0:t0_cutoff_idx_discharge] # NEL discharged
    elif region == 'Homerton':
        population = data_pop.iloc[0,2] #population
        # PHE cases data
        cases = data_cases.iloc[t0_start_idx_PHE:t0_cutoff_idx_PHE,3]
        # NHSE deaths data (NEL only)
        deaths = data_deaths.CumHomerton[:t0_cutoff_idx_deaths] # NEL cumulative deaths                 
        # NHSE/I Covid-19 Dashboard sitrep data (NEL only)
        # hospitalised 
        hospitalised = data_sitrep_hosp.Homerton[0:t0_cutoff_idx_hosp] # NEL hospitalised
        ICU = data_sitrep_hosp.Homerton_ITU[0:t0_cutoff_idx_hosp] # NEL hospitalised (critical care)
        # discharged
        discharged = data_sitrep_discharge.CumHomerton[0:t0_cutoff_idx_discharge] # NEL discharged
    
    
    #%% Define objective function that least squares will try to optimise = residual between model and actuals
    # Optimiser will try to minimise the output of this function
    def objective(x):       
        # lmfit params
        R0 = x['R0']
        t0 = x['t0']
        InterventionTime2 = InterventionTime2_day-t0
        InterventionTime = InterventionTime1_day-t0
        InterventionAmt = x['InterventionAmt']
        InterventionAmt2 = x['InterventionAmt2']
        
#        D_incubation = x['D_incubation']
#        D_infectious = x['D_infectious']
        
        D_hospital_lag_severe = x['D_hospital_lag_severe']
        D_hospital_lag_fatal = x['D_hospital_lag_fatal']
        D_recovery_severe = x['D_recovery_severe']
        D_death = x['D_death']
        
#        P_SEVERE = x['P_SEVERE']
        ICU_prop = x['ICU_prop']
        CFR = x['CFR']
        
        SEIR_table = SEIRfun.SEIR_results(t,population,ICU_prop,
                                          InterventionTime,InterventionTime2,InterventionAmt,InterventionAmt2,R0,D_incubation,D_infectious,
                                          P_SEVERE,CFR,D_recovery_mild,D_hospital_lag_severe,D_hospital_lag_fatal,D_recovery_severe,D_death)
            
        t_shift = t + t0
        
        print('fit=',x)
        
        # residual 1: PHE cases data vs. Predicted total cases
        cases_proj = SEIR_table[:,cases_col_idx]
        model_interp = interp1d(t_shift, cases_proj, kind='cubic',fill_value="extrapolate")
        model_interp_cases = model_interp(cases_date) # interpolate to match model x-axis values to case data
        resid1 = (model_interp_cases-cases)#/model_interp_cases # residual between model and actuals
        
        # residual 2: Deaths data vs. Predicted cumulative deaths
        deaths_proj = SEIR_table[:,fatality_cum_col_idx]
        model_interp = interp1d(t_shift, deaths_proj, kind='cubic',fill_value="extrapolate")
        model_interp_deaths = model_interp(date_deaths) # interpolate to match model x-axis values to case data
        resid2 = (model_interp_deaths-deaths)#/model_interp_deaths # residual between model and actuals
        
        # residual 3: Hospitalised data vs. Predicted hospitalised 
        hosp_proj = SEIR_table[:,hosp_col_idx]
        model_interp = interp1d(t_shift, hosp_proj, kind='cubic',fill_value="extrapolate")
        model_interp_hosp = model_interp(date_hosp) # interpolate to match model x-axis values to case data
        resid3 = (model_interp_hosp-hospitalised)#/model_interp_hosp # residual between model and actuals
        
        # residual 4: ICU data vs. Predicted ICU 
        ICU_proj = SEIR_table[:,ICU_col_idx]
        model_interp = interp1d(t_shift, ICU_proj, kind='cubic',fill_value="extrapolate")
        model_interp_ICU = model_interp(date_hosp) # interpolate to match model x-axis values to case data
        resid4 = (model_interp_ICU-ICU)#/model_interp_ICU # residual between model and actuals
        
        # residual 5: Discharged data vs. Predicted discharged 
        discharge_proj = SEIR_table[:,discharge_cum_col_idx]
        model_interp = interp1d(t_shift, discharge_proj, kind='cubic',fill_value="extrapolate")
        model_interp_discharge = model_interp(date_discharge) # interpolate to match model x-axis values to case data
        resid5 = (model_interp_discharge-discharged)#/model_interp_discharge # residual between model and actuals
                    
        #obj = resid2 # fit to single parameter
        obj = np.concatenate((resid2,resid3,resid4,resid5)) # residual between model and actuals. concatenate to get flat column with all residuals in one
        return obj
            
    #%% create a set of Parameters and their inital values and bounds
    params = Parameters()
    params.add('R0', value=3.2, min=1.5, max=4.0)
    params.add('t0', value=43877.0, min=43860.0, max=43899.0)
    params.add('InterventionAmt', value=0.7, min=0.4, max=1.0)
    params.add('InterventionAmt2', value=0.25, min=0.1, max=0.6)
    
#    params.add('D_incubation', value=5, min=1, max=10)
#    params.add('D_infectious', value=1, min=0.5, max=10)
    
    params.add('D_hospital_lag_severe', value=7, min=0.1, max=20)
    params.add('D_hospital_lag_fatal', value=5, min=0.01, max=20)
    params.add('D_recovery_severe', value=7, min=0.5, max=30)
    params.add('D_death', value=7, min=0.5, max=30)
    
#    params.add('P_SEVERE', value=0.044, min=0.01, max=0.2)
    params.add('ICU_prop', value=0.25, min=0.15, max=0.4)
    params.add('CFR', value=0.01, min=0.005, max=0.05)
  
    #%% Do fitting
    fit_method = 'leastsq' # choose fit method. Options: differential_evolution, leastsq, basinhopping, ampgo, shgo, dual_annealing
    fit = minimize(objective, params, method = fit_method) # run minimisation
    print(fit_report(fit)) # print fitting results

    #%% Define fitted parameters
    x = fit.params
    R0 = x['R0'].value
    t0 = x['t0'].value
    InterventionTime2 = InterventionTime2_day-t0
    InterventionTime = InterventionTime1_day-t0
    InterventionAmt = x['InterventionAmt'].value
    InterventionAmt2 = x['InterventionAmt2'].value
#        D_incubation = x['D_incubation'].value
#        D_infectious = x['D_infectious'].value
    D_hospital_lag_severe = x['D_hospital_lag_severe'].value
    D_hospital_lag_fatal = x['D_hospital_lag_fatal'].value
    D_recovery_severe = x['D_recovery_severe'].value
    D_death = x['D_death'].value
#        P_SEVERE = x['P_SEVERE'].value
    ICU_prop = x['ICU_prop'].value
    CFR = x['CFR'].value
    
    params_fit = []
    params_fit.append(R0)
    params_fit.append(t0)
    params_fit.append(InterventionTime1_day)
    params_fit.append(InterventionAmt)
    params_fit.append(InterventionAmt2)
    
#    params_fit.append(D_incubation)
#    params_fit.append(D_infectious)
    params_fit.append(D_hospital_lag_severe)
    params_fit.append(D_hospital_lag_fatal)
    params_fit.append(D_recovery_severe)
    params_fit.append(D_death)

    params_fit.append(ICU_prop)
    params_fit.append(CFR)
#    params_fit.append(P_SEVERE)

    
    #%% Generate data using all fitted parameters
    t_shift = t + t0
    xfit = SEIRfun.SEIR_results(t,population,ICU_prop,
                                      InterventionTime,InterventionTime2,InterventionAmt,InterventionAmt2,R0,D_incubation,D_infectious,
                                      P_SEVERE,CFR,D_recovery_mild,D_hospital_lag_severe,D_hospital_lag_fatal,D_recovery_severe,D_death)
    
    # interpolate so that all model results for all regions are eqivalent
    interp = interp1d(t_shift, xfit, kind='cubic', axis = 0, fill_value="extrapolate")
    x_interpolated = interp(t_final) 
    
    # Re-calculate daily figures after interpolation, in order to get daily changes
    x_interpolated[1:,discharge_daily_col_idx] = np.diff(x_interpolated[:,discharge_cum_col_idx]) # rate of change in discharged patients, based on stepwise increase in recovered from hospital
    x_interpolated[0,discharge_daily_col_idx] = 0 # insert 0 as first value in daily column
    x_interpolated[1:,fatality_daily_col_idx] = np.diff(x_interpolated[:,fatality_cum_col_idx]) # rate of change in fatalities, from stepwise increase in deaths
    x_interpolated[0,fatality_daily_col_idx] = 0 # insert 0 as first value in daily column
    
  
    #%% Plot predicted against actuals
    plt.figure()
    plt.plot(t_final, x_interpolated[:,hosp_col_idx], '-b') # hospitalised daily
    plt.plot(t_final, x_interpolated[:,ICU_col_idx], '-r') # ICU daily
    plt.plot(t_final, x_interpolated[:,cases_col_idx], '-y') # total cases
    plt.plot(t_final, x_interpolated[:,discharge_cum_col_idx], '-g') # recovered from severe
    plt.plot(t_final, x_interpolated[:,fatality_cum_col_idx], '-k') # deaths
    if region == 'NEL':
        plt.plot(data_cases.date, data_cases.iloc[:,1], color='gold', marker='o', linestyle='dashed')
        plt.plot(data_deaths.Date, data_deaths.CumBHRUTBartsHom, color='black', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.BHRUT_Barts_Hom, color='blue', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.BHRUT_Barts_Hom_ITU, color='red', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_discharge.Date, data_sitrep_discharge.CumBHRUTBartsHom, color='green', marker='o', linestyle='dashed')
    elif region == 'BHRUT': 
        plt.plot(data_cases.date, data_cases.iloc[:,9], color='gold', marker='o', linestyle='dashed')
        plt.plot(data_deaths.Date, data_deaths.CumBHRUT, color='black', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.BHRUT, color='blue', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.BHRUT_ITU, color='red', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_discharge.Date, data_sitrep_discharge.CumBHRUT, color='green', marker='o', linestyle='dashed')
    elif region == 'Barts':
        plt.plot(data_cases.date, data_cases.iloc[:,10], color='gold', marker='o', linestyle='dashed')
        plt.plot(data_deaths.Date, data_deaths.CumBarts, color='black', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.Barts, color='blue', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.Barts_ITU, color='red', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_discharge.Date, data_sitrep_discharge.CumBarts, color='green', marker='o', linestyle='dashed')
    elif region == 'Homerton':
        plt.plot(data_cases.date, data_cases.iloc[:,3], color='gold', marker='o', linestyle='dashed')
        plt.plot(data_deaths.Date, data_deaths.CumHomerton, color='black', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.Homerton, color='blue', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.Homerton_ITU, color='red', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_discharge.Date, data_sitrep_discharge.CumHomerton, color='green', marker='o', linestyle='dashed')

    plt.legend(['hospitalised (pred)', 'ICU (pred)','total cases (pred)', 'discharged (pred)', 'deaths (pred)', 'PHE cases data','NHSE cumulative deaths','NHSEI hospitalised','NHSEI ITU','NHSEI discharged'], loc='best')
    plt.show() 
    
    
    #%% summary stats
    SEIR_table = x_interpolated
    stats = np.zeros([11])
    # size and date of daily hospitalisation peak
    stats[0] = max(SEIR_table[:,hosp_col_idx]) # size 
    stats[1] = t_final[np.where(SEIR_table[:,hosp_col_idx] == max(SEIR_table[:,hosp_col_idx]))] #  date
    # size and date of daily ICU hospitalisation peak
    stats[2] = max(SEIR_table[:,ICU_col_idx]) # size 
    stats[3] = t_final[np.where(SEIR_table[:,ICU_col_idx] == max(SEIR_table[:,ICU_col_idx]))] #  date
    # size and date of daily discharges peak
    stats[4] = max(SEIR_table[:,discharge_daily_col_idx]) # size 
    stats[5] = t_final[np.where(SEIR_table[:,discharge_daily_col_idx] == max(SEIR_table[:,discharge_daily_col_idx]))] #  date
    # size and date of daily deaths peak
    stats[6] = max(SEIR_table[:,fatality_daily_col_idx]) # size 
    stats[7] = t_final[np.where(SEIR_table[:,fatality_daily_col_idx] == max(SEIR_table[:,fatality_daily_col_idx]))] #  date
    
    stats[8] = max(SEIR_table[:,discharge_cum_col_idx]) # cumulative number of discharges (recovered from severe)
    stats[9] = max(SEIR_table[:,fatality_cum_col_idx]) # cumulative number of deaths
    stats[10] = stats[8] + stats[9] # cumulative number of hospitalised (recovered(severe (hospitalised)) + deaths)

        
        
#%% Produce final data tables and headline stats
stats_list = []
SEIR_table_list = []
pop_idx = [0, 8, 9, 2] # index locations of relevant populations 

# Generate final SEIR tables for different regions
for n in range(0,fit_params.shape[1]-1,1): # for each region
    print(n)
    population = data_pop.iloc[0,pop_idx[n]] #population
    
    # Define parameters obtained from fitting
    R0 = fit_params.iloc[0,n+1]
    t0 = fit_params.iloc[1,n+1]
    InterventionTime2 = InterventionTime2_day-t0
    InterventionTime = InterventionTime1_day-t0
    InterventionAmt = fit_params.iloc[3,n+1]
    InterventionAmt2 = fit_params.iloc[4,n+1]
    D_hospital_lag_severe = fit_params.iloc[5,n+1]
    D_hospital_lag_fatal = fit_params.iloc[6,n+1]
    D_recovery_severe = fit_params.iloc[7,n+1]
    D_death = fit_params.iloc[8,n+1]
    ICU_prop = fit_params.iloc[9,n+1]
    CFR = fit_params.iloc[10,n+1]
    P_SEVERE = fit_params.iloc[11,n+1]
    
    # Generate data using fitted parameters
    t_shift = t + t0
    xfit = SEIRfun.SEIR_results(t,population,ICU_prop,
                                      InterventionTime,InterventionTime2,InterventionAmt,InterventionAmt2,R0,D_incubation,D_infectious,
                                      P_SEVERE,CFR,D_recovery_mild,D_hospital_lag_severe,D_hospital_lag_fatal,D_recovery_severe,D_death)
    
    # interpolate to 't_final' so that all model results for all regions are eqivalent
    interp = interp1d(t_shift, xfit, kind='cubic', axis = 0, fill_value="extrapolate")
    x_interpolated = interp(t_final) 
    
    # Re-calculate daily figures after interpolation, in order to get daily changes
    x_interpolated[1:,discharge_daily_col_idx] = np.diff(x_interpolated[:,discharge_cum_col_idx]) # rate of change in discharged patients, based on stepwise increase in recovered from hospital
    x_interpolated[0,discharge_daily_col_idx] = 0 # insert 0 as first value in daily column
    x_interpolated[1:,fatality_daily_col_idx] = np.diff(x_interpolated[:,fatality_cum_col_idx]) # rate of change in fatalities, from stepwise increase in deaths
    x_interpolated[0,fatality_daily_col_idx] = 0 # insert 0 as first value in daily column
    
    SEIR_table_list.append(x_interpolated) 
    
SEIR_table_list.append(SEIR_table_list[1] + SEIR_table_list[2] + SEIR_table_list[3]) # NEL sum of 3 trusts

# Generate Summary Stats
for n in range(0,len(SEIR_table_list),1): # for each region    
    # summary stats
    stats = np.zeros([11])
    # size and date of daily hospitalisation peak
    stats[0] = max(SEIR_table_list[n][:,hosp_col_idx]) # size 
    stats[1] = t_final[np.where(SEIR_table_list[n][:,hosp_col_idx] == max(SEIR_table_list[n][:,hosp_col_idx]))] #  date
    # size and date of daily ICU hospitalisation peak
    stats[2] = max(SEIR_table_list[n][:,ICU_col_idx]) # size 
    stats[3] = t_final[np.where(SEIR_table_list[n][:,ICU_col_idx] == max(SEIR_table_list[n][:,ICU_col_idx]))] #  date
    # size and date of daily discharges peak
    stats[4] = max(SEIR_table_list[n][:,discharge_daily_col_idx]) # size 
    stats[5] = t_final[np.where(SEIR_table_list[n][:,discharge_daily_col_idx] == max(SEIR_table_list[n][:,discharge_daily_col_idx]))] #  date
    # size and date of daily deaths peak
    stats[6] = max(SEIR_table_list[n][:,fatality_daily_col_idx]) # size 
    stats[7] = t_final[np.where(SEIR_table_list[n][:,fatality_daily_col_idx] == max(SEIR_table_list[n][:,fatality_daily_col_idx]))] #  date
    
    stats[8] = max(SEIR_table_list[n][:,discharge_cum_col_idx]) # cumulative number of discharges (recovered from severe)
    stats[9] = max(SEIR_table_list[n][:,fatality_cum_col_idx]) # cumulative number of deaths
    stats[10] = stats[8] + stats[9] # cumulative number of hospitalised (recovered(severe (hospitalised)) + deaths)
      
    stats_list.append(stats)
    
    
# Create borough level breakdowns by applying proportion of PHE confirmed cases by borough to NEL (sum of 3 trust fits)
stats_list_borough = []
SEIR_table_list_borough = []
# BHR
SEIR_table_list_borough.append((max(data_cases.Barking)/max(data_cases.NEL)) * SEIR_table_list[-1])
SEIR_table_list_borough.append((max(data_cases.Havering)/max(data_cases.NEL)) * SEIR_table_list[-1])
SEIR_table_list_borough.append((max(data_cases.Redbridge)/max(data_cases.NEL)) * SEIR_table_list[-1])
# WEL
SEIR_table_list_borough.append((max(data_cases.Newham)/max(data_cases.NEL)) * SEIR_table_list[-1])
SEIR_table_list_borough.append((max(data_cases.TowerHamlets)/max(data_cases.NEL)) * SEIR_table_list[-1])
SEIR_table_list_borough.append((max(data_cases.WalthamForest)/max(data_cases.NEL)) * SEIR_table_list[-1])
# City & Hackney
SEIR_table_list_borough.append((max(data_cases.Hackney)/max(data_cases.NEL)) * SEIR_table_list[-1])

# Generate borough level summary stats
for n in range(0,len(SEIR_table_list_borough),1): # for each region    
    # summary stats
    stats = np.zeros([11])
    # size and date of daily hospitalisation peak
    stats[0] = max(SEIR_table_list_borough[n][:,hosp_col_idx]) # size 
    stats[1] = t_final[np.where(SEIR_table_list_borough[n][:,hosp_col_idx] == max(SEIR_table_list_borough[n][:,hosp_col_idx]))] #  date
    # size and date of daily ICU hospitalisation peak
    stats[2] = max(SEIR_table_list_borough[n][:,ICU_col_idx]) # size 
    stats[3] = t_final[np.where(SEIR_table_list_borough[n][:,ICU_col_idx] == max(SEIR_table_list_borough[n][:,ICU_col_idx]))] #  date
    # size and date of daily discharges peak
    stats[4] = max(SEIR_table_list_borough[n][:,discharge_daily_col_idx]) # size 
    stats[5] = t_final[np.where(SEIR_table_list_borough[n][:,discharge_daily_col_idx] == max(SEIR_table_list_borough[n][:,discharge_daily_col_idx]))] #  date
    # size and date of daily deaths peak
    stats[6] = max(SEIR_table_list_borough[n][:,fatality_daily_col_idx]) # size 
    stats[7] = t_final[np.where(SEIR_table_list_borough[n][:,fatality_daily_col_idx] == max(SEIR_table_list_borough[n][:,fatality_daily_col_idx]))] #  date
    
    stats[8] = max(SEIR_table_list_borough[n][:,discharge_cum_col_idx]) # cumulative number of discharges (recovered from severe)
    stats[9] = max(SEIR_table_list_borough[n][:,fatality_cum_col_idx]) # cumulative number of deaths
    stats[10] = stats[8] + stats[9] # cumulative number of hospitalised (recovered(severe (hospitalised)) + deaths)
    
    stats_list_borough.append(stats)
    
    
    