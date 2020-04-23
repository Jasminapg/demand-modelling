# Based on: https://github.com/gabgoh/epcalc/blob/master/src/App.svelte
# SEIR model 25/03/2020

#%% import packages
import numpy as np
from scipy import optimize as opt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import SEIR_model_functions_15_04_20 as SEIRfun

import copy
from lmfit import Minimizer, Parameters, fit_report, minimize

#%% Load data
# CONNECT TO FEED FROM POWER BI USING API?
# load case data
#data_cases = pd.read_excel("NEL_cases.xlsx") # old form of reporting
data_cases = pd.read_excel("NEL_cases_new.xlsx") # new form of reporting from Wed 15th April - reporting by day of test
# load ONS population data
data_pop = pd.read_excel("NEL_pop_ons.xlsx") # ONS mid-year 2019
# Load population by age data
data_pop_age = pd.read_excel("NEL_pop_age.xlsx") # Population by age
# Load hospitalisation by age data (Hospitalisation and ICU from Imperial study, 16/03 and subsequent Lancet paper, Verity et al.)
#data_hosp_age = pd.read_excel("NEL_hosp_age.xlsx") 
# Load hospitalisation by age data (Hospitalisation from Imperial study... , ICU flat average from NEL sitrep data)
data_hosp_age = pd.read_excel("NEL_hosp_age_sitrep.xlsx") 

# Load NHSE deaths (NEL only)
data_deaths = pd.read_excel("NEL_deaths.xlsx")

# Load NHSE/I Covid19 dashboard sitrep (NEL only)
data_sitrep_hosp = pd.read_excel("NEL_sitrep_hospitalisation.xlsx") 
data_sitrep_discharge = pd.read_excel("NEL_sitrep_discharge.xlsx") 

# Load fitted parameters
fit_params = np.loadtxt('model_fitted_parameters_2inter_1304_mod.csv', delimiter = ',')


#%% Fixed model inputs
## Transmission dynamics inputs
# Population
N                 = 2043143 # population
I0                = 1 # number of initial infections
# Reproduction number
R0                = 3.3 # initial R0 value
# Transmission times
D_incubation      = 5.1 # incubation time (days)
D_infectious      = 1 # infectious time (days)

## Clinical dynamics inputs
# morbidity statistics
CFR               = 0.01 # fraction of cases that are fatal, case fatality ratio
Time_to_death     = 17.8 # 17.8 time from end of incubation period to death
D_death           = Time_to_death - D_infectious 
# recovery times
D_recovery_mild   = 7  #- D_infectious # recovery time for mild patients
D_recovery_severe = 3 # 3, 10.4 - D_infectious # recovery time for severe patients (hospital length of stay)

# care statistics
P_SEVERE          = 0.044 # fraction of cases that are severe (require hospitalisation)
D_hospital_lag    = 7 # 7 time from symptoms to hospital
ICU_prop          = 0.23

## Intervention inputs
InterventionTime  = 25  # intervention day
OMInterventionAmt = 0.30 # effect of intervention in reducing transmission
InterventionAmt   = 1 - OMInterventionAmt
# 2nd Intervention 
InterventionTime2  = 32  # intervention day
OMInterventionAmt2 = 0.50 # effect of intervention in reducing transmission
InterventionAmt2   = 1 - OMInterventionAmt2

#%% Other inputs
# time axis for fitting
Time              = 220.0 # number of days in the model 
dt                = 0.25 # time step (days)
duration          = 7*12*1e10
t = np.linspace(0.0,Time,int((Time+dt)/dt)) # time points

# Column headings 
col_names = ['Susceptible','Exposed','Infectious','Mild','Severe (pre-hospital)','Hospitalised (severe)','Hospitalised (fatal)'
                 ,'Recovered (mild)','Discharged (cumulative)','Deaths (cumulative)'
                 , 'Recovered (all)', 'Hospitalised (all)', 'Hospitalised (ICU)', 'Discharged (daily)'
                 , 'Deaths (daily)', 'Total confirmed cases']


#%% Variables to control what code does:
intervention_2 = 2 # 0 for not accounting for 2nd intervention (lockdown), 1 for FIXED effect of lockdown, 2 to FIT effect of lockdown
intervention2_fraction = 0.1 # set fixed transmission proportion (vs. original R0) after 2nd intervention

do_fitting = 1 # = 1 to get code to do the fitting. if not, skips to collation of results based on model fitting parameters file

fit_dates = data_cases.date.shape[0] # data_cases.date.shape[0] for full date range
regions = 1#data_cases.shape[1]-1 # choose how many regions to run fitting for. to run all regions: data_cases.shape[1]-1

cases_col_idx = 15 # pick which column to fit actual data to - 11 = hospitalised, 15 = PHE equivalent
hosp_col_idx =  11 # column index of hospitalised column
ICU_col_idx = 12 # column index of ICU column
discharge_daily_col_idx = 13 # idx of daily discharges col
discharge_cum_col_idx = 8 # idx of cumulative discharged col
fatality_daily_col_idx = 14 # idx of daily deaths col
fatality_cum_col_idx = 9 # idx of cumulative deaths col

# create time axis to use for all results
time_min = np.ceil(max(fit_params[:,1])) # first day of model as the highest t0 for all regions, rounded up to nearest integer
time_max = np.floor(Time + min(fit_params[0:regions,1])) # min(fit_params[:,1]) # final day of model as the lowest t0 + number of days, rounded down to nearest integer
dt_final = 1 # time step (days)
t_final = np.linspace(time_min,time_max,int((time_max-time_min+dt_final)/dt_final)) # final time axis, to use for all results

# create empty tables/lists to fill with final data
SEIR_table = np.zeros([t_final.shape[0],len(col_names),regions])

t0_fit_start = 43896
t0_cutoff = data_sitrep_hosp.Date.iloc[-5] # data_cases.date.iloc[-2] or 43908. cut off date so no data after more severe interventions began is included 
InterventionTime2_day = 43914
InterventionTime1_day = 43907
# 43902 = Thurs 12th March - Self-isolation guidelines
# 43906 = Mon 16th March - Day 1 of Social Distancing measures
# 43910 = Fri 20th March - Announcement of school and pub closures
# 43913 = Mon 23rd March - Day 1 of UK lockdown

if do_fitting == 1: 
    t0_start_idx_PHE = np.where(data_cases.date == t0_fit_start)[0][0]
    t0_cutoff_idx_PHE = np.where(data_cases.date == t0_cutoff+1)[0][0]
    cases_date = data_cases.date[t0_start_idx_PHE:t0_cutoff_idx_PHE]
    
    t0_cutoff_idx_deaths = np.where(data_deaths.Date == t0_cutoff+1)[0][0]
    date_deaths = data_deaths.Date[0:t0_cutoff_idx_deaths]
    
    t0_cutoff_idx_hosp = np.where(data_sitrep_hosp.Date == t0_cutoff+1)[0][0]
    date_hosp = data_sitrep_hosp.Date[0:t0_cutoff_idx_hosp]
    
    t0_cutoff_idx_discharge = np.where(data_sitrep_discharge.Date == t0_cutoff+1)[0][0]
    date_discharge = data_sitrep_discharge.Date[0:t0_cutoff_idx_discharge]
    
    for n in range(0,regions,1): # for each region 
        
#        if n > 1: # for other regions, use NEL fits for the variables that were previously fixed
#            # fix values for times, and maybe CFR, P_SEVERE and ICU_prop
        
        print('n = ',n)
        
        population = data_pop.iloc[0,n] #population
        
        # PHE cases data
        cases = data_cases.iloc[t0_start_idx_PHE:t0_cutoff_idx_PHE,n+1]
                
        # NHSE deaths data (NEL only)
        deaths = data_deaths.Cum_BHRUT_Barts_Hom[:t0_cutoff_idx_deaths] # NEL cumulative deaths         
        
        # NHSE/I Covid-19 Dashboard sitrep data (NEL only)
        # hospitalised 
        hospitalised = data_sitrep_hosp.Hospitalised_BHRUT_Barts_Hom[0:t0_cutoff_idx_hosp] # NEL hospitalised
        ICU = data_sitrep_hosp.ITU_BHRUT_Barts_Hom[0:t0_cutoff_idx_hosp] # NEL hospitalised (critical care)
        # discharged
        discharged = data_sitrep_discharge.DischargedCumulative_BHRUT_Barts_Hom[0:t0_cutoff_idx_discharge] # NEL discharged
        
    
        #%% Define objective function that least squares will try to optimise = residual between model and actuals
        # Optimiser will try to minimise the output of this function
        def objective(x):
            #%% FITTING STAGE 1: Find best fit for R0 and t0, for no intervention, up to date of first IRL intervention, for each borough ###
    
            #%% Fitting part 1
            # set transmission rates = 1 and intervention time = 0 to show no intervention
#            InterventionAmt = 1 
#            InterventionTime = 0 # set = 0, arbitrary
#            InterventionAmt2 = 1 # set = 1 to show no intervention
#            InterventionTime2 = 0 # set = 0, arbitrary
            
            # lmfit params
            R0 = x['R0']
            t0 = x['t0']
            InterventionTime2 = InterventionTime2_day-t0
            InterventionTime = InterventionTime1_day-t0
#            t1 = x['InterventionTime']
#            InterventionTime = t1-t0
            InterventionAmt = x['InterventionAmt']
            InterventionAmt2 = x['InterventionAmt2']
            D_incubation = x['D_incubation']
            D_infectious = x['D_infectious']
            D_hospital_lag = x['D_hospital_lag']
            D_recovery_severe = x['D_recovery_severe']
            D_death = x['D_death']
            P_SEVERE = x['P_SEVERE']
#            ICU_prop = x['ICU_prop']
            CFR = x['CFR']
            
            SEIR_table = SEIRfun.SEIR_results(t,population,ICU_prop,
                                              InterventionTime,InterventionTime2,InterventionAmt,InterventionAmt2,R0,D_incubation,D_infectious,
                                              P_SEVERE,CFR,D_recovery_mild,D_hospital_lag,D_recovery_severe,D_death)
                
#            SEIR_table = SEIRfun.SEIR_subpop_sum(t,col_names,data_pop,data_pop_age,data_hosp_age,n,
#                                                 InterventionTime,InterventionTime2,InterventionAmt,InterventionAmt2,R0,D_incubation,D_infectious,
#                                                 P_SEVERE,CFR,D_recovery_mild,D_hospital_lag,D_recovery_severe,D_death)
            
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
                        
            #obj = resid3 # fit to single parameter
            obj = np.concatenate((resid2,resid3,resid4,resid5)) # residual between model and actuals
            #obj = np.concatenate((resid1,resid2,resid3,resid4,resid5)) # residual between model and actuals

            return obj
        
        # 2 params
#        bnds = ((0,43880.0),(5,43899.0)) # set bounds for variables being fitted
#        x0 = [3,43890] # set initial conditions
        
        # 5 params
#        bnds = ((0,43880.0,43906.0,0.1,0.1),(5,43899.0,43907.0,0.9,0.7)) # set bounds for variables being fitted
#        x0 = [3,43890,43906.5,0.7,0.3] # set initial conditions

        # 10 params
#        bnds = ((0,43880.0,43906.0,0.1,0.1,0,0,0,0,0),(5,43899.0,43907.0,0.9,0.7,14,14,14,21,28)) # set bounds for variables being fitted
#        x0 = [3,43890,43906.5,0.7,0.3,5.1,1,7,7,17.8] # set initial conditions

        # 13 params
        bnds = ((0,43880.0,43906.0,0.1,0.1,0,0,0,0,0,0,0,0),(5,43899.0,43907.0,0.9,0.7,14,14,14,21,28,0.07,0.3,0.05)) # set bounds for variables being fitted
        x0 = [3,43890,43906.5,0.7,0.3,5.1,1,7,7,17.8,0.044,0.2,0.01] # set initial conditions
        
        # Fitting using lmfit
        # create a set of Parameters
        params = Parameters()
        params.add('R0', value=3.6, min=1.5, max=4.0)
        params.add('t0', value=43894.0, min=43860.0, max=43899.0)
#        params.add('InterventionTime', value=43906.0, min=43906.0, max=43907.0)
        params.add('InterventionAmt', value=0.7, min=0.5, max=1.0)
        params.add('InterventionAmt2', value=0.3, min=0.0, max=0.6)
        
        params.add('D_incubation', value=5, min=1, max=10)
        params.add('D_infectious', value=1, min=0.5, max=10)
        params.add('D_hospital_lag', value=3, min=1, max=7)
        params.add('D_recovery_severe', value=9, min=8, max=10)
        params.add('D_death', value=11, min=9, max=17)
        params.add('P_SEVERE', value=0.044, min=0.01, max=0.2)
#        params.add('ICU_prop', value=0.23, min=0.2, max=0.27)
        params.add('CFR', value=0.01, min=0.005, max=0.05)
        
        # set step size
#        params['R0'].set(brute_step = 0.2)
#        params['t0'].set(brute_step = 1)
##        params['InterventionTime'].set(brute_step = 0.2)
#        params['InterventionAmt'].set(brute_step = 1)
#        params['InterventionAmt2'].set(brute_step = 1)
#        params['D_incubation'].set(brute_step = 1)
#        params['D_infectious'].set(brute_step = 1)
#        params['D_hospital_lag'].set(brute_step = 1)
#        params['D_recovery_severe'].set(brute_step = 1)
#        params['D_death'].set(brute_step = 2)
#        params['P_SEVERE'].set(brute_step = 0.025)
#        params['ICU_prop'].set(brute_step = 0.05)
#        params['CFR'].set(brute_step = 0.01)
        
        
        fit_method = 'leastsq' # differential_evolution, leastsq, basinhopping, ampgo, shgo, dual_annealing
        fit = minimize(objective, params, method = fit_method)
        print(fit_report(fit))
        

#        fitter = Minimizer(objective, params)#,nan_policy='omit')
#        result_brute = fitter.minimize(method='brute', Ns=2 keep=25)
#        print(fit_report(result_brute))
#        
#        # and is, therefore, probably not so likely... However, as said above, in most
#        # cases you'll want to do another minimization using the solutions from the
#        # ``brute`` method as starting values. That can be easily accomplished as
#        # shown in the code below, where we now perform a ``leastsq`` minimization
#        # starting from the top-25 solutions and accept the solution if the ``chisqr``
#        # is lower than the previously 'optimal' solution:
#        best_result = copy.deepcopy(result_brute)
#
#        for candidate in result_brute.candidates:
#            trial = fitter.minimize(method='leastsq', params=candidate.params)
#            if trial.chisqr < best_result.chisqr:
#                best_result = trial
#        
#        # From the ``leastsq`` minimization we obtain the following parameters for the most optimal result:
#        print(fit_report(best_result))
#        fit = best_result
        
        x = fit.params
        R0 = x['R0'].value
        t0 = x['t0'].value
        InterventionTime2 = InterventionTime2_day-t0
        InterventionTime = InterventionTime1_day-t0
#        t1 = x['InterventionTime'].value
#        InterventionTime = t1-t0
        InterventionAmt = x['InterventionAmt'].value
        InterventionAmt2 = x['InterventionAmt2'].value
        D_incubation = x['D_incubation'].value
        D_infectious = x['D_infectious'].value
        D_hospital_lag = x['D_hospital_lag'].value
        D_recovery_severe = x['D_recovery_severe'].value
        D_death = x['D_death'].value
        P_SEVERE = x['P_SEVERE'].value
#        ICU_prop = x['ICU_prop'].value
        CFR = x['CFR'].value
        
        params_fit = []
        params_fit.append(R0)
        params_fit.append(t0)
#        params_fit.append(t1)
        params_fit.append(InterventionAmt)
        params_fit.append(InterventionAmt2)
        params_fit.append(D_incubation)
        params_fit.append(D_infectious)
        params_fit.append(D_hospital_lag)
        params_fit.append(D_recovery_severe)
        params_fit.append(D_death)
        params_fit.append(P_SEVERE)
#        params_fit.append(ICU_prop)
        params_fit.append(CFR)
    
        
        # Generate data using all fitted parameters
        t_shift = t + t0
        
        xfit = SEIRfun.SEIR_results(t,population,ICU_prop,
                                          InterventionTime,InterventionTime2,InterventionAmt,InterventionAmt2,R0,D_incubation,D_infectious,
                                          P_SEVERE,CFR,D_recovery_mild,D_hospital_lag,D_recovery_severe,D_death)
        
#        xfit = SEIRfun.SEIR_subpop_sum(t,col_names,data_pop,data_pop_age,data_hosp_age,n,
#                                      InterventionTime,InterventionTime2,InterventionAmt,InterventionAmt2,R0,D_incubation,D_infectious,
#                                      P_SEVERE,CFR,D_recovery_mild,D_hospital_lag,D_recovery_severe,D_death)
        
        # interpolate so that all model results for all regions are eqivalent
        interp = interp1d(t_shift, xfit, kind='cubic', axis = 0)
        x_interpolated = interp(t_shift) 
        
        # Re-calculate daily figures after interpolation, in order to get daily changes
        x_interpolated[1:,discharge_daily_col_idx] = np.diff(x_interpolated[:,8]) # rate of change in discharged patients, based on stepwise increase in recovered from hospital
        x_interpolated[0,discharge_daily_col_idx] = 0 # insert 0 as first value in daily column
        x_interpolated[1:,fatality_daily_col_idx] = np.diff(x_interpolated[:,9]) # rate of change in fatalities, from stepwise increase in deaths
        x_interpolated[0,fatality_daily_col_idx] = 0 # insert 0 as first value in daily column
        
#        SEIR_table[:,:,n] = x_interpolated       
    
        plt.figure()
        plt.plot(t_shift, x_interpolated[:,hosp_col_idx], '-b') # hospitalised daily
        plt.plot(t_shift, x_interpolated[:,ICU_col_idx], '-r') # ICU daily
        plt.plot(t_shift, x_interpolated[:,cases_col_idx], '-y') # total cases
        plt.plot(t_shift, x_interpolated[:,8], '-g') # recovered from severe
        plt.plot(t_shift, x_interpolated[:,9], '-k') # deaths
        plt.plot(data_cases.date, data_cases.iloc[:,n+1], color='gold', marker='o', linestyle='dashed')
        plt.plot(data_deaths.Date, data_deaths.Cum_BHRUT_Barts_Hom, color='black', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.Hospitalised_BHRUT_Barts_Hom, color='blue', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.ITU_BHRUT_Barts_Hom, color='red', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_discharge.Date, data_sitrep_discharge.DischargedCumulative_BHRUT_Barts_Hom, color='green', marker='o', linestyle='dashed')
        plt.legend(['hospitalised (pred)', 'ICU (pred)','total cases (pred)', 'discharged (pred)', 'deaths (pred)', 'PHE cases data','NHSE cumulative deaths','NHSEI hospitalised','NHSEI ITU','NHSEI discharged'], loc='best')
        plt.show() 
        
        
    
#    #%% FITTING STAGE 2: Now do the same, but fitting actual data to-date, and varying reduction in transmission rate
#    # This time, keep t0 and R0 (before intervention) fixed, using values from previous fitting
#    
#    for n in range(0,regions,1): # for each region    
#        R0_fit = fit_params[n,0]
#        t_shift = t + fit_params[n,1]
#        inter1_effect_fit = fit_params[n,2]
#        InterventionTime_fit = fit_params[n,3]-fit_params[n,1]
#        inter2_effect_fit = fit_params[n,4]
#        InterventionTime2 = 43914-fit_params[n,1] # tuesday 24th march - day after lockdown commenced
#        
#        # PHE cases data
#        date_full = data_cases.date[:fit_dates]
#        cases_full = data_cases.iloc[:fit_dates,n+1]
#        population = data_pop.iloc[0,n]
#        
#        # NHSE deaths data (NEL only)
#        tshift_0_idx = np.where(data_deaths.Date >= t_shift[0])[0][0]
#        date_deaths = data_deaths.Date[tshift_0_idx:]
#        deaths = data_deaths.NELcumulative[tshift_0_idx:] # NEL cumulative deaths         
#    
#        # NHSE/I Covid-19 Dashboard sitrep data (NEL only)
#        # hospitalised 
#        tshift_0_idx = np.where(data_sitrep_hosp.Date >= t_shift[0])[0][0]
#        date_hosp = data_sitrep_hosp.Date[tshift_0_idx:]
#        hospitalised = data_sitrep_hosp.Hospitalised[tshift_0_idx:] # NEL hospitalised
#        ICU = data_sitrep_hosp.ITU[tshift_0_idx:] # NEL hospitalised (critical care)
#        # discharged
#        tshift_0_idx = np.where(data_sitrep_discharge.Date >= t_shift[0])[0][0]
#        date_discharge = data_sitrep_discharge.Date[tshift_0_idx:]
#        discharged = data_sitrep_discharge.DischargedCumulative[tshift_0_idx:] # NEL discharged
#        
#        #%% Define objective function that least squares will try to optimise = residual between model and actuals
#        # Optimiser will try to minimise the output of this function
#        def objective(x):
#            InterventionAmt1 = x[0]
#            InterventionAmt2 = x[1]
#            InterventionTime = x[2]-fit_params[n,1]
#            #D_death = x[3]
#            #R0 = x[4]
#            
#            SEIR_table = SEIRfun.SEIR_subpop_sum(t,col_names,data_pop,data_pop_age,data_hosp_age,n,
#                                                InterventionTime,InterventionTime2,InterventionAmt1,InterventionAmt2,R0_fit,D_incubation,D_infectious,
#                                                P_SEVERE,CFR,D_recovery_mild,D_hospital_lag,D_recovery_severe,D_death)
#            #SEIR_table = SEIRfun.SEIR_subpop_sum(t,R0_fit,InterventionTime_test,InterventionTime2,InterventionAmt1_test,InterventionAmt2_test,col_names,data_pop,data_pop_age,data_hosp_age,n)
#            
#            # residual 1: PHE cases data vs. Predicted total cases
#            cases_proj = SEIR_table[:,cases_col_idx]
#            model_interp = interp1d(t_shift, cases_proj, kind='cubic',fill_value="extrapolate")
#            model_interp_cases = model_interp(date_full) # interpolate to match model x-axis values to case data
#            resid1 = (model_interp_cases-cases_full)/model_interp_cases # residual between model and actuals
#            
#            # residual 2: Deaths data vs. Predicted cumulative deaths
#            deaths_proj = SEIR_table[:,fatality_cum_col_idx]
#            model_interp = interp1d(t_shift, deaths_proj, kind='cubic',fill_value="extrapolate")
#            model_interp_deaths = model_interp(date_deaths) # interpolate to match model x-axis values to case data
#            resid2 = (model_interp_deaths-deaths)/model_interp_deaths # residual between model and actuals
#            
#            # residual 3: Hospitalised data vs. Predicted hospitalised 
#            hosp_proj = SEIR_table[:,hosp_col_idx]
#            model_interp = interp1d(t_shift, hosp_proj, kind='cubic',fill_value="extrapolate")
#            model_interp_hosp = model_interp(date_hosp) # interpolate to match model x-axis values to case data
#            resid3 = (model_interp_hosp-hospitalised)/model_interp_hosp # residual between model and actuals
#            
#            # residual 4: Discharged data vs. Predicted discharged 
#            discharge_proj = SEIR_table[:,discharge_cum_col_idx]
#            model_interp = interp1d(t_shift, discharge_proj, kind='cubic',fill_value="extrapolate")
#            model_interp_discharge = model_interp(date_discharge) # interpolate to match model x-axis values to case data
#            resid4 = (model_interp_discharge-discharged)/model_interp_discharge # residual between model and actuals
#            
#            #obj = resid1 # fit to single parameter
#            obj = np.concatenate((resid1,resid2,resid3,resid4)) # residual between model and actuals
#            return obj
#        
#        bnds = ((0.1,0.1,43906.0),(0.9,0.7,43907.0)) # set bounds for variables being fitted
#        x0 = [0.7,0.3,43906.5] # set initial conditions
#
#        res = opt.least_squares(objective, x0, bounds = bnds, xtol = None) # do least squares optimisation
#        print('n = ',n,', answer', res)
#        
#        # revise fitted parameters after fitting
#        InterventionAmt1 = res['x'][0]
#        InterventionAmt2 = res['x'][1]
#        InterventionTime = res['x'][2]-fit_params[n,1]
#        #D_death = res['x'][3]
#        #R0_fit = res['x'][4]
#        
#        #fit_params[n,0] = res['x'][4]
#        fit_params[n,2] = res['x'][0]
#        fit_params[n,4] = res['x'][1]
#        fit_params[n,3] = res['x'][2]
#
#        # Generate data using all fitted parameters
#        x = SEIRfun.SEIR_subpop_sum(t,col_names,data_pop,data_pop_age,data_hosp_age,n,
#                                    InterventionTime,InterventionTime2,InterventionAmt1,InterventionAmt2,R0_fit,D_incubation,D_infectious,
#                                    P_SEVERE,CFR,D_recovery_mild,D_hospital_lag,D_recovery_severe,D_death)
#        
#        #x = SEIRfun.SEIR_subpop_sum(t,R0_fit,InterventionTime_fit,InterventionTime2,inter1_effect_fit,inter2_effect_fit,col_names,data_pop,data_pop_age,data_hosp_age,n)     
#        
#        # interpolate so that all model results for all regions are eqivalent
#        interp = interp1d(t_shift, x, kind='cubic', axis = 0)
#        x_interpolated = interp(t_final) 
#        
#        # Re-calculate daily figures after interpolation, in order to get daily changes
#        x_interpolated[1:,discharge_daily_col_idx] = np.diff(x_interpolated[:,8]) # rate of change in discharged patients, based on stepwise increase in recovered from hospital
#        x_interpolated[0,discharge_daily_col_idx] = 0 # insert 0 as first value in daily column
#        x_interpolated[1:,fatality_daily_col_idx] = np.diff(x_interpolated[:,9]) # rate of change in fatalities, from stepwise increase in deaths
#        x_interpolated[0,fatality_daily_col_idx] = 0 # insert 0 as first value in daily column
#        
#        SEIR_table[:,:,n] = x_interpolated       
#    
#        plt.figure()
#        plt.plot(t_final, SEIR_table[:,hosp_col_idx,n], '-b') # hospitalised daily
#        plt.plot(t_final, SEIR_table[:,ICU_col_idx,n], '-r') # ICU daily
#        plt.plot(t_final, SEIR_table[:,cases_col_idx,n], '-y') # total cases
#        plt.plot(t_final, SEIR_table[:,8,n], '-g') # recovered from severe
#        plt.plot(t_final, SEIR_table[:,9,n], '-k') # deaths
#        plt.plot(data_cases.date, data_cases.iloc[:,n+1], color='gold', marker='o', linestyle='dashed')
#        plt.plot(date_deaths, deaths, color='black', marker='o', linestyle='dashed')
#        plt.plot(date_hosp, hospitalised, color='blue', marker='o', linestyle='dashed')
#        plt.plot(date_discharge, discharged, color='green', marker='o', linestyle='dashed')
#        plt.legend(['hospitalised (pred)', 'ICU (pred)','total cases (pred)', 'discharged (pred)', 'deaths (pred)', 'PHE cases data','NHSE cumulative deaths','NHSEI hospitalised','NHSEI discharged'], loc='best')
#        plt.show() 