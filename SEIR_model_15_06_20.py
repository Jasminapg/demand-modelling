# SEIR model 19/05/2020
# WEL Financial Strategy Team, WEL CCGs (Tower Hamlets, Newham, Waltham Forest CCGs)
# Written by: Nathan Cheetham, Senior Financial Strategy Analyst, nathan.cheetham1@nhs.net
# Adapted from: https://github.com/gabgoh/epcalc/blob/master/src/App.svelte

# Script fits SEIR parameters to best replicate observed data on hospitalisation, discharges and deaths
# in North East London (NEL) hospital trusts (Barts Health, BHRUT, Homerton UT)

#%% import packages
import datetime
import numpy as np
from scipy import optimize as opt
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import SEIR_model_functions_15_06_20 as SEIRfun
from sklearn.model_selection import train_test_split
import pandas as pd
import copy
from lmfit import Minimizer, Parameters, fit_report, minimize

font = {'size'   : 17}
plt.rc('font', **font)

#%% Variables to control what code does:
do_fitting = 0 # = 1 to get code to do the fitting. if not, skips to collation of results based on model fitting parameters file
plot_results = 1 # = 1 to plot and get stats for fitting results
do_scenarios = 0 # = 1 to do scenarios with varying R and date of easing

region = 'NEL' # choose data to fit to. Option: NEL, Barts, BHRUT, Homerton

region_list = ['NEL','BHRUT','Barts','Homerton','NEL trust sum']


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
fit_params = pd.read_csv('model_fitted_parameters_1006.csv', delimiter = ',')


#%% SEIR Model input parameters
I0                = 1 # number of initial infections
D_incubation      = 5.1 # incubation time (days)
D_infectious      = 1 # infectious time (days)
D_infectious_asym = 1 # infectious time for asymptomatic cases (days)
p_sym = 0.7 # proportion of cases that are symptomatic

# CURRENTLY FITTED
# Reproduction number prior to intervention
R0                = 3.2 # effective R0
#R0_sym                = 3.2 # initial R0 value for symptomatic cases
#R0_asym               = 3.2 # initial R0 value for asymptomatic cases
#R0 = p_sym*R0_sym + (1-p_sym)*R0_asym # effective R0

## Clinical dynamics inputs
# recovery times
D_hospital_lag_GA   = 7 # time from symptoms to hospital for GA cases
D_hospital_lag_ICU   = 7 # time from symptoms to hospital for ICU cases
D_recovery_asym = 7 # recovery time for asymptomatic patients
D_recovery_mild = 7 # recovery time for mild patients
D_recovery_GA = 6 # BHRUT: 6 recovery time for non-ICU patients (hospital length of stay)
D_recovery_ICU = 13 # BHRUT: 13 recovery time for non-ICU patients (hospital length of stay)
D_death_GA = 6 # BHRUT: 6 recovery time for non-ICU patients (hospital length of stay) - assume same for death as recovery
D_death_ICU = 13 # BHRUT: 13 recovery time for non-ICU patients (hospital length of stay) - assume same for death as recovery

# care statistics
P_SEVERE          = 0.044 # fraction of cases that are severe (require hospitalisation)
CFR_GA = 0.33 # Fatality rate for G&A hospitalised cases. 33% from BHRUT data
CFR_ICU = 0.5 # Fatality rate for ICU hospitalised cases. 50% from BHRUT data
p_ICU          = 0.2 # Proportion of hospitalised cases requiring ICU/ITU bed

# time axis for fitting
Time              = 365.0 # number of days in the model 
dt                = 0.25 # time step (days)
duration          = 7*12*1e10
t = np.linspace(0.0,Time,int((Time+dt)/dt)) # time points

# create time axis to use for all results
time_min = np.ceil(max(fit_params.iloc[3,1:])) # 43880. first day of model as the highest t0 for all regions, rounded up to nearest integer
time_max = np.floor(Time + min(fit_params.iloc[3,1:])) #time_min + Time - 10 # min(fit_params[:,1]) # final day of model as the lowest t0 + number of days, rounded down to nearest integer
dt_final = 1 # time step (days)
t_final = np.linspace(time_min,time_max,int((time_max-time_min+dt_final)/dt_final)) # final time axis, to use for all results

time_min_dateformat = datetime.datetime(2020, 2, 17, 00, 00) # 43878 = 17/02/2020
t_final_dateformat = pd.date_range(time_min_dateformat, periods=Time-1).tolist()
dates = matplotlib.dates.date2num(t_final_dateformat)

t0_fit_start = 43896 # start date of actual data for fitting
t0_cutoff = data_sitrep_hosp.Date.iloc[-3] # cut off date for actual data for fitting

## Intervention inputs
# 43902 = Thurs 12th March - Self-isolation guidelines
# 43906 = Mon 16th March - Social Distancing measures announced
# 43910 = Fri 20th March - Announcement of school and pub closures
# 43913 = Mon 23rd March - UK lockdown announced
# 43961 = Sun 10th May - Plan for easing announced. 
# 43979 = Thursday 29th June - PLan for 1st June (43983) school reopening and other easing, expanded social contact, outdoor markets etc

# Set dates of interventions based on UK measures
InterventionTime1_day = 43907
InterventionTime2_day = 43914
# 1st Intervention
InterventionAmt   = 0.7 # effect of intervention in reducing transmission = fractional reduction in beta_hat
# 2nd Intervention 
InterventionAmt2   = 0.23 # effect of intervention in reducing transmission = fractional reduction in beta_hat
# Hypothetical interventions for easing lockdown:
InterventionTime3_day = 43950 # Empirical to give best fit. 45000 or 43946 # 43961 = Sun 10th May - Plan for easing announced. 
if plot_results == 1 or do_fitting == 1:
    InterventionTime4_day = 45000 # Day beyond time range, so effectively no effect on model
if do_scenarios == 1:
    InterventionTime4_day = 43979 # Day of easing announcement, for scenarios
InterventionAmt3 = 3.2/3.2 # effect of intervention in reducing transmission = fractional reduction in beta_hat
InterventionAmt4 = 6 / 10.6 # effect of intervention in reducing transmission = fractional reduction in beta_hat


# Column headings 
col_names = ['Susceptible','Exposed','Infectious (symptomatic)','Infectious (asymptomatic)','Asymptomatic','Mild','Pre-hospital','Hospitalised (GA)','Hospitalised (ICU)'
                 ,'Recovered (asymptomatic)','Recovered (mild)','Discharged (cumulative)','Deaths (cumulative)'
                 , 'Recovered (all)', 'Hospitalised (all)', 'Discharged (daily)'
                 , 'Deaths (daily)', 'Total confirmed cases','Effective beta','Beta','Contacts, c','R_t']


# column indices for relevant columns for plotting
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
idx_R = 23


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
        deaths = data_deaths.CumBHRUTBartsHom[:t0_cutoff_idx_deaths] # cumulative deaths                 
        deaths_daily = data_deaths.DailyBHRUTBartsHom[:t0_cutoff_idx_deaths] # daily deaths  
        # NHSE/I Covid-19 Dashboard sitrep data (NEL only)
        # hospitalised 
        hospitalised = data_sitrep_hosp.BHRUT_Barts_Hom[0:t0_cutoff_idx_hosp] # NEL hospitalised
        GA = data_sitrep_hosp.BHRUT_Barts_Hom_GA[0:t0_cutoff_idx_hosp] # NEL hospitalised (GA)
        ICU = data_sitrep_hosp.BHRUT_Barts_Hom_ITU[0:t0_cutoff_idx_hosp] # NEL hospitalised (critical care)
        # discharged
        discharged = data_sitrep_discharge.CumBHRUTBartsHom[0:t0_cutoff_idx_discharge] # NEL discharged
        discharged_daily = data_sitrep_discharge.DailyBHRUTBartsHom[0:t0_cutoff_idx_discharge] # NEL discharged
    elif region == 'BHRUT':
        population = data_pop.iloc[0,8] #population
        # PHE cases data
        cases = data_cases.iloc[t0_start_idx_PHE:t0_cutoff_idx_PHE,9]
        # NHSE deaths data (NEL only)
        deaths = data_deaths.CumBHRUT[:t0_cutoff_idx_deaths] # cumulative deaths                 
        deaths_daily = data_deaths.DailyBHRUT[:t0_cutoff_idx_deaths] # daily deaths 
        # NHSE/I Covid-19 Dashboard sitrep data (NEL only)
        # hospitalised 
        hospitalised = data_sitrep_hosp.BHRUT[0:t0_cutoff_idx_hosp] # NEL hospitalised
        GA = data_sitrep_hosp.BHRUT_GA[0:t0_cutoff_idx_hosp] # NEL hospitalised (GA)
        ICU = data_sitrep_hosp.BHRUT_ITU[0:t0_cutoff_idx_hosp] # NEL hospitalised (critical care)
        # discharged
        discharged = data_sitrep_discharge.CumBHRUT[0:t0_cutoff_idx_discharge] # Cumulative discharged
        discharged_daily = data_sitrep_discharge.DailyBHRUT[0:t0_cutoff_idx_discharge] # Daily discharged
    elif region == 'Barts':
        population = data_pop.iloc[0,9] #population
        # PHE cases data
        cases = data_cases.iloc[t0_start_idx_PHE:t0_cutoff_idx_PHE,10]
        # NHSE deaths data (NEL only)
        deaths = data_deaths.CumBarts[:t0_cutoff_idx_deaths] # cumulative deaths   
        deaths_daily = data_deaths.DailyBarts[:t0_cutoff_idx_deaths] # daily deaths               
        # NHSE/I Covid-19 Dashboard sitrep data (NEL only)
        # hospitalised 
        hospitalised = data_sitrep_hosp.Barts[0:t0_cutoff_idx_hosp] # NEL hospitalised
        GA = data_sitrep_hosp.Barts_GA[0:t0_cutoff_idx_hosp] # NEL hospitalised (GA)
        ICU = data_sitrep_hosp.Barts_ITU[0:t0_cutoff_idx_hosp] # NEL hospitalised (critical care)
        # discharged
        discharged = data_sitrep_discharge.CumBarts[0:t0_cutoff_idx_discharge] # Cumulative discharged
        discharged_daily = data_sitrep_discharge.DailyBarts[0:t0_cutoff_idx_discharge] # Daily discharged
    elif region == 'Homerton':
        population = data_pop.iloc[0,2] #population
        # PHE cases data
        cases = data_cases.iloc[t0_start_idx_PHE:t0_cutoff_idx_PHE,3]
        # NHSE deaths data (NEL only)
        deaths = data_deaths.CumHomerton[:t0_cutoff_idx_deaths] # NEL cumulative deaths
        deaths_daily = data_deaths.DailyHomerton[:t0_cutoff_idx_deaths] # daily deaths                  
        # NHSE/I Covid-19 Dashboard sitrep data (NEL only)
        # hospitalised 
        hospitalised = data_sitrep_hosp.Homerton[0:t0_cutoff_idx_hosp] # NEL hospitalised
        GA = data_sitrep_hosp.Homerton_GA[0:t0_cutoff_idx_hosp] # NEL hospitalised (GA)
        ICU = data_sitrep_hosp.Homerton_ITU[0:t0_cutoff_idx_hosp] # NEL hospitalised (critical care)
        # discharged
        discharged = data_sitrep_discharge.CumHomerton[0:t0_cutoff_idx_discharge] # Cumulative discharged
        discharged_daily = data_sitrep_discharge.DailyHomerton[0:t0_cutoff_idx_discharge] # Daily discharged
    
    #%% Use parameters obtained from fitting the '2 intervention' scenario
#    if region == 'NEL':
#        n = 0
#    elif region == 'BHRUT':
#        n = 1
#    elif region == 'Barts':
#        n = 2
#    elif region == 'Homerton':
#        n =3
#        
#    R0 = fit_params.iloc[2,n+1]
#    t0 = fit_params.iloc[3,n+1]
#    InterventionTime = InterventionTime1_day-t0
#    InterventionTime2 = InterventionTime2_day-t0
#    #InterventionTime3_day = fit_params.iloc[18,n+1]
#    InterventionTime3 = InterventionTime3_day-t0
#    InterventionTime4 = InterventionTime4_day-t0
#    InterventionAmt = fit_params.iloc[4,n+1]
#    InterventionAmt2 = fit_params.iloc[5,n+1]
#    InterventionAmt3 = fit_params.iloc[19,n+1]
#    D_hospital_lag_GA = fit_params.iloc[6,n+1]
#    D_hospital_lag_ICU = fit_params.iloc[7,n+1]
#    D_recovery_GA = fit_params.iloc[8,n+1]
#    D_recovery_ICU = fit_params.iloc[9,n+1]
#    D_death_GA = fit_params.iloc[10,n+1]
#    D_death_ICU = fit_params.iloc[11,n+1]
#    P_SEVERE = fit_params.iloc[12,n+1]
#    p_ICU = fit_params.iloc[13,n+1]
#    CFR_GA = fit_params.iloc[14,n+1]
#    CFR_ICU = fit_params.iloc[15,n+1]
#    D_infectious_asym = fit_params.iloc[16,n+1]
#    p_sym = fit_params.iloc[17,n+1]
    
    
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
               
        D_hospital_lag_GA = x['D_hospital_lag_GA']
        D_hospital_lag_ICU = x['D_hospital_lag_ICU']
        D_recovery_GA = x['D_recovery_GA']
        D_recovery_ICU = x['D_recovery_ICU']
        D_death_GA = x['D_death_GA']
        D_death_ICU = x['D_death_ICU']
        
        P_SEVERE = x['P_SEVERE']
        p_ICU = x['p_ICU']
        CFR_GA = x['CFR_GA']
        CFR_ICU = x['CFR_ICU']

        D_infectious_asym = x['D_infectious_asym']
        p_sym = x['p_sym']

#        D_incubation = x['D_incubation']
#        D_infectious = x['D_infectious']

        # Further interventions
        InterventionAmt3 = x['InterventionAmt3']
#        InterventionTime3_day = x['InterventionTime3_day']
        InterventionTime3 = InterventionTime3_day-t0
        InterventionTime4 = InterventionTime4_day-t0
        #InterventionAmt4 = x['InterventionAmt4']
                
        SEIR_table = SEIRfun.SEIR_results(t,population,
                                          InterventionTime,InterventionTime2,InterventionTime3,InterventionTime4,InterventionAmt,InterventionAmt2,InterventionAmt3,InterventionAmt4,R0,D_incubation,D_infectious,D_infectious_asym,
                                          P_SEVERE,CFR_GA,CFR_ICU,p_ICU,p_sym,D_recovery_asym,D_recovery_mild,D_hospital_lag_GA,D_hospital_lag_ICU,D_recovery_GA,D_recovery_ICU,D_death_GA,D_death_ICU)

        t_shift = t + t0
        
        print('fit=',x)
        
        # split observed data into training and testing data
#        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#        t_train = 
        # Minimise residual between model and training data
        
        # Measure score of model on test data
        
        # Calculate residuals for peak normalised data, so each residual has ~ roughly equal weighting
        # residual 1: PHE cases data vs. Predicted total cases
        cases_proj = SEIR_table[:,idx_cases]
        model_interp = interp1d(t_shift, cases_proj, kind='cubic',fill_value="extrapolate")
        model_interp_cases = model_interp(cases_date) # interpolate to match model x-axis values to case data
        resid1 = (model_interp_cases-cases)#/model_interp_cases # residual between model and actuals
        
        # residual 2: Cumulative Deaths data vs. Predicted cumulative deaths
        deaths_proj = SEIR_table[:,idx_R_fatal]
        model_interp = interp1d(t_shift, deaths_proj, kind='cubic',fill_value="extrapolate")
        model_interp_deaths = model_interp(date_deaths) # interpolate to match model x-axis values to case data
        model_interp_deaths = model_interp_deaths / max(deaths)
        deaths_norm = deaths / max(deaths)
        resid2 = (model_interp_deaths-deaths_norm)#/model_interp_deaths # residual between model and actuals
        
        # residual 2b: Daily Deaths data vs. Predicted daily deaths
        deaths_proj = SEIR_table[:,fatality_daily_col_idx]
        model_interp = interp1d(t_shift, deaths_proj, kind='cubic',fill_value="extrapolate")
        model_interp_deaths_daily = model_interp(date_deaths) # interpolate to match model x-axis values to case data
        resid2b = (model_interp_deaths_daily-deaths_daily)#/model_interp_deaths_daily # residual between model and actuals
        
        # residual 3: Hospitalised data vs. Predicted hospitalised 
        hosp_proj = SEIR_table[:,hosp_col_idx]
        model_interp = interp1d(t_shift, hosp_proj, kind='cubic',fill_value="extrapolate")
        model_interp_hosp = model_interp(date_hosp) # interpolate to match model x-axis values to case data
        model_interp_hosp = model_interp_hosp / max(hospitalised)
        hospitalised_norm = hospitalised / max(hospitalised)
        resid3 = (model_interp_hosp-hospitalised_norm)#/model_interp_hosp # residual between model and actuals
        
        # residual 4: GA data vs. Predicted GA 
        GA_proj = SEIR_table[:,idx_hospitalised_GA]
        model_interp = interp1d(t_shift, GA_proj, kind='cubic',fill_value="extrapolate")
        model_interp_GA = model_interp(date_hosp) # interpolate to match model x-axis values to case data
        model_interp_GA = model_interp_GA / max(GA)
        GA_norm = GA / max(GA)
        resid4 = (model_interp_GA-GA_norm)#/model_interp_GA # residual between model and actuals
        
        # residual 5: ICU data vs. Predicted ICU 
        ICU_proj = SEIR_table[:,idx_hospitalised_ICU]
        model_interp = interp1d(t_shift, ICU_proj, kind='cubic',fill_value="extrapolate")
        model_interp_ICU = model_interp(date_hosp) # interpolate to match model x-axis values to case data
        model_interp_ICU = model_interp_ICU / max(ICU)
        ICU_norm = ICU / max(ICU)
        resid5 = (model_interp_ICU-ICU_norm)#/model_interp_ICU # residual between model and actuals
        
        # residual 6: Cumulative Discharged data vs. Predicted discharged cumulative
        discharge_proj = SEIR_table[:,idx_R_hospitalised]
        model_interp = interp1d(t_shift, discharge_proj, kind='cubic',fill_value="extrapolate")
        model_interp_discharge = model_interp(date_discharge) # interpolate to match model x-axis values to case data
        model_interp_discharge = model_interp_discharge / max(discharged)
        discharged_norm = discharged / max(discharged)
        resid6 = (model_interp_discharge-discharged_norm)#/model_interp_discharge # residual between model and actuals
        
        # residual 6b: Daily Discharged data vs. Predicted discharged daily
        discharge_proj = SEIR_table[:,discharge_daily_col_idx]
        model_interp = interp1d(t_shift, discharge_proj, kind='cubic',fill_value="extrapolate")
        model_interp_discharge_daily = model_interp(date_discharge) # interpolate to match model x-axis values to case data
        resid6b = (model_interp_discharge_daily-discharged_daily)#/model_interp_discharge_daily # residual between model and actuals
                    
        #obj = resid3 # fit to single parameter
        obj = np.concatenate((resid2,resid3,resid4,resid5,resid6)) # residual between model and actuals. concatenate to get flat column with all residuals in one
        return obj
            
    #%% create a set of Parameters and their inital values and bounds
    params = Parameters()
    params.add('R0', value=3.2, min=2.8, max=3.5)
    params.add('t0', value=43877.0, min=43860.0, max=43899.0)
    params.add('InterventionAmt', value=0.69, min=0.4, max=1.0)
    params.add('InterventionAmt2', value=0.2, min=0.1, max=0.6)
     
    params.add('D_hospital_lag_GA', value=7, min=0.1, max=20)
    params.add('D_hospital_lag_ICU', value=7, min=0.1, max=20)
    params.add('D_recovery_GA', value=6, min=2, max=12)
    params.add('D_recovery_ICU', value=13, min=8, max=20)
    params.add('D_death_GA', value=6, min=2, max=12)
    params.add('D_death_ICU', value=13, min=8, max=20)
    
    params.add('P_SEVERE', value=0.044, min=0.01, max=0.2)
    params.add('p_ICU', value=0.18, min=0.15, max=0.25)
    params.add('CFR_GA', value=0.26, min=0.15, max=0.35)
    params.add('CFR_ICU', value=0.5, min=0.4, max=0.6)

#    params.add('D_incubation', value=5, min=1, max=10)
#    params.add('D_infectious', value=1, min=0.5, max=10)
    params.add('D_infectious_asym', value=1, min=0.5, max=20)
    params.add('p_sym', value=0.7, min=0.3, max=1)
    
    params.add('InterventionAmt3', value=0.25, min=0.01, max=0.6)
#    params.add('InterventionTime3_day', value= 43947.0, min=43921.0, max=43967.0)
#    params.add('InterventionAmt4', value=0.25, min=0.01, max=0.9)
    
    #%% Do fitting
    fit_method = 'leastsq' # choose fit method. Options: differential_evolution, leastsq, basinhopping, ampgo, shgo, dual_annealing
    fit = minimize(objective, params, method = fit_method) # run minimisation
    print(fit_report(fit)) # print fitting results

    #%% Define fitted parameters
    x = fit.params
    R0 = x['R0'].value
    t0 = x['t0'].value
    
    InterventionTime = InterventionTime1_day-t0
    InterventionTime2 = InterventionTime2_day-t0
    InterventionAmt = x['InterventionAmt'].value
    InterventionAmt2 = x['InterventionAmt2'].value
    
#    InterventionTime3_day = x['InterventionTime3_day'].value
    InterventionAmt3 = x['InterventionAmt3'].value
#    InterventionAmt4 = x['InterventionAmt4'].value
    
    D_hospital_lag_GA = x['D_hospital_lag_GA'].value
    D_hospital_lag_ICU = x['D_hospital_lag_ICU'].value
    D_recovery_GA = x['D_recovery_GA'].value
    D_recovery_ICU = x['D_recovery_ICU'].value
    D_death_GA = x['D_death_GA'].value
    D_death_ICU = x['D_death_ICU'].value
    P_SEVERE = x['P_SEVERE'].value
    
    p_ICU = x['p_ICU'].value
    CFR_GA = x['CFR_GA'].value
    CFR_ICU = x['CFR_ICU'].value
    
    D_infectious_asym = x['D_infectious_asym'].value
    p_sym = x['p_sym'].value    
    
#    D_incubation = x['D_incubation'].value
#    D_infectious = x['D_infectious'].value
    
    #R0 = p_sym*R0_sym + (1-p_sym)*R0_asym
    gamma = 1/D_infectious
    gamma_asym = 1/D_infectious_asym
    beta_0 = R0*(1/((p_sym/gamma) + ((1-p_sym)/gamma_asym)))
    R0_sym = beta_0/gamma
    R0_asym = beta_0/gamma_asym
    
    params_fit = []
    params_fit.append(R0_sym)
    params_fit.append(R0_asym)
    params_fit.append(R0)
    params_fit.append(t0)
    params_fit.append(InterventionAmt)
    params_fit.append(InterventionAmt2)

    
    params_fit.append(D_hospital_lag_GA)
    params_fit.append(D_hospital_lag_ICU)
    params_fit.append(D_recovery_GA)
    params_fit.append(D_recovery_ICU)
    params_fit.append(D_death_GA)
    params_fit.append(D_death_ICU)
    params_fit.append(P_SEVERE)
    
    params_fit.append(p_ICU)
    params_fit.append(CFR_GA)
    params_fit.append(CFR_ICU)

#    params_fit.append(D_incubation)
#    params_fit.append(D_infectious)
    params_fit.append(D_infectious_asym)
    params_fit.append(p_sym)
    
#    params_fit.append(InterventionTime3_day)
    params_fit.append(InterventionAmt3)
#    params_fit.append(InterventionAmt4)
    print(params_fit)
    
    
    
    #%% Now score each 'folds' fit to training data, based on how well they fit the test data (rather than the lowest residual considering all data at once)
    # TO DOOO
    # residual between model and test data for each fold
    
    # calculate average residual
    # minimise this average residual which measures how model compares against testing data (residual). 
    
    # pick the fitting parameters that give the smallest residual with test data
    
    # min_idx = np.where(residual == min(residual))
    # params_fit_best = param_fit(min()
    

    
    #%% Generate data using all fitted parameters
    t_shift = t + t0
    # Hypothetical interventions
    InterventionTime3 = InterventionTime3_day-t0
    InterventionTime4 = InterventionTime4_day-t0
    
    xfit = SEIRfun.SEIR_results(t,population,
                                      InterventionTime,InterventionTime2,InterventionTime3,InterventionTime4,InterventionAmt,InterventionAmt2,InterventionAmt3,InterventionAmt4,R0,D_incubation,D_infectious,D_infectious_asym,
                                      P_SEVERE,CFR_GA,CFR_ICU,p_ICU,p_sym,D_recovery_asym,D_recovery_mild,D_hospital_lag_GA,D_hospital_lag_ICU,D_recovery_GA,D_recovery_ICU,D_death_GA,D_death_ICU)
    
    # interpolate so that all model results for all regions are eqivalent
    interp = interp1d(t_shift, xfit, kind='cubic', axis = 0, fill_value="extrapolate")
    x_interpolated = interp(t_final) 
    
    # Re-calculate daily figures after interpolation, in order to get daily changes
    x_interpolated[1:,discharge_daily_col_idx] = np.diff(x_interpolated[:,idx_R_hospitalised]) # rate of change in discharged patients, based on stepwise increase in recovered from hospital
    x_interpolated[0,discharge_daily_col_idx] = 0 # insert 0 as first value in daily column
    x_interpolated[1:,fatality_daily_col_idx] = np.diff(x_interpolated[:,idx_R_fatal]) # rate of change in fatalities, from stepwise increase in deaths
    x_interpolated[0,fatality_daily_col_idx] = 0 # insert 0 as first value in daily column
    
    
    
    
    
  
    #%% Plot predicted against actuals
    plt.figure()
    plt.plot(t_final, x_interpolated[:,hosp_col_idx], '-b') # hospitalised daily
    plt.plot(t_final, x_interpolated[:,idx_hospitalised_GA], '-y') # GA daily
    plt.plot(t_final, x_interpolated[:,idx_hospitalised_ICU], '-r') # ICU daily
    plt.plot(t_final, x_interpolated[:,idx_cases], '-y') # total cases
    plt.plot(t_final, x_interpolated[:,idx_R_hospitalised], '-g') # discharged cumulative
    plt.plot(t_final, x_interpolated[:,discharge_daily_col_idx], '-g') # discharged daily
    plt.plot(t_final, x_interpolated[:,idx_R_fatal], '-k') # deaths
    plt.plot(t_final, x_interpolated[:,fatality_daily_col_idx], '-k') # deaths daily
    if region == 'NEL':
        plt.plot(data_cases.date, data_cases.iloc[:,1], color='gold', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.BHRUT_Barts_Hom, color='blue', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.BHRUT_Barts_Hom_GA, color='yellow', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.BHRUT_Barts_Hom_ITU, color='red', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_discharge.Date, data_sitrep_discharge.CumBHRUTBartsHom, color='green', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_discharge.Date, data_sitrep_discharge.DailyBHRUTBartsHom, color='green', marker='o', linestyle='dashed')
        plt.plot(data_deaths.Date, data_deaths.CumBHRUTBartsHom, color='black', marker='o', linestyle='dashed')
        plt.plot(data_deaths.Date, data_deaths.DailyBHRUTBartsHom, color='black', marker='o', linestyle='dashed')
    elif region == 'BHRUT': 
        plt.plot(data_cases.date, data_cases.iloc[:,9], color='gold', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.BHRUT, color='blue', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.BHRUT_GA, color='yellow', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.BHRUT_ITU, color='red', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_discharge.Date, data_sitrep_discharge.CumBHRUT, color='green', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_discharge.Date, data_sitrep_discharge.DailyBHRUT, color='green', marker='o', linestyle='dashed')
        plt.plot(data_deaths.Date, data_deaths.CumBHRUT, color='black', marker='o', linestyle='dashed')
        plt.plot(data_deaths.Date, data_deaths.DailyBHRUT, color='black', marker='o', linestyle='dashed')
    elif region == 'Barts':
        plt.plot(data_cases.date, data_cases.iloc[:,10], color='gold', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.Barts, color='blue', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.Barts_GA, color='yellow', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.Barts_ITU, color='red', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_discharge.Date, data_sitrep_discharge.CumBarts, color='green', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_discharge.Date, data_sitrep_discharge.DailyBarts, color='green', marker='o', linestyle='dashed')
        plt.plot(data_deaths.Date, data_deaths.CumBarts, color='black', marker='o', linestyle='dashed')
        plt.plot(data_deaths.Date, data_deaths.DailyBarts, color='black', marker='o', linestyle='dashed')
    elif region == 'Homerton':
        plt.plot(data_cases.date, data_cases.iloc[:,3], color='gold', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.Homerton, color='blue', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.Homerton_GA, color='yellow', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_hosp.Date, data_sitrep_hosp.Homerton_ITU, color='red', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_discharge.Date, data_sitrep_discharge.CumHomerton, color='green', marker='o', linestyle='dashed')
        plt.plot(data_sitrep_discharge.Date, data_sitrep_discharge.DailyHomerton, color='green', marker='o', linestyle='dashed')
        plt.plot(data_deaths.Date, data_deaths.CumHomerton, color='black', marker='o', linestyle='dashed')
        plt.plot(data_deaths.Date, data_deaths.DailyHomerton, color='black', marker='o', linestyle='dashed')
        
    plt.xlabel('Date')
    plt.legend(['hospitalised (pred)', 'GA (pred)', 'ICU (pred)','total cases (pred)', 'cum discharged (pred)', 'daily discharged (pred)', 'deaths (pred)', 'daily deaths (pred)', 'PHE cases data','NHSEI hospitalised','NHSEI GA','NHSEI ITU','NHSEI cum discharged','NHSEI daily discharged','NHSE cumulative deaths','NHSE daily deaths'], loc='best')
    plt.show() 
    
    
    #%% summary stats
    SEIR_table = x_interpolated
    stats = np.zeros([11])
    # size and date of daily hospitalisation peak
    stats[0] = max(SEIR_table[:,hosp_col_idx]) # size 
    stats[1] = t_final[np.where(SEIR_table[:,hosp_col_idx] == max(SEIR_table[:,hosp_col_idx]))] #  date
    # size and date of daily ICU hospitalisation peak
    stats[2] = max(SEIR_table[:,idx_hospitalised_ICU]) # size 
    stats[3] = t_final[np.where(SEIR_table[:,idx_hospitalised_ICU] == max(SEIR_table[:,idx_hospitalised_ICU]))] #  date
    # size and date of daily discharges peak
    stats[4] = max(SEIR_table[:,discharge_daily_col_idx]) # size 
    stats[5] = t_final[np.where(SEIR_table[:,discharge_daily_col_idx] == max(SEIR_table[:,discharge_daily_col_idx]))] #  date
    # size and date of daily deaths peak
    stats[6] = max(SEIR_table[:,fatality_daily_col_idx]) # size 
    stats[7] = t_final[np.where(SEIR_table[:,fatality_daily_col_idx] == max(SEIR_table[:,fatality_daily_col_idx]))] #  date
    
    stats[8] = max(SEIR_table[:,idx_R_hospitalised]) # cumulative number of discharges (recovered from severe)
    stats[9] = max(SEIR_table[:,idx_R_fatal]) # cumulative number of deaths
    stats[10] = stats[8] + stats[9] # cumulative number of hospitalised (recovered(severe (hospitalised)) + deaths)

        
        
#%% Produce final data tables and headline stats
    
if plot_results == 1:
    
    stats_list = []
    SEIR_table_list = []
    pop_idx = [0, 8, 9, 2] # index locations of relevant populations 
    
    # Generate final SEIR tables for different regions
    for n in range(0,fit_params.shape[1]-1,1): # for each region
        print(n)
        population = data_pop.iloc[0,pop_idx[n]] #population
        
        # Define parameters obtained from fitting
        R0 = fit_params.iloc[2,n+1]
        t0 = fit_params.iloc[3,n+1]
        InterventionTime = InterventionTime1_day-t0
        InterventionTime2 = InterventionTime2_day-t0
        InterventionTime3_day = fit_params.iloc[18,n+1]
        InterventionTime3 = InterventionTime3_day-t0
        InterventionTime4 = InterventionTime4_day-t0
        InterventionAmt = fit_params.iloc[4,n+1]
        InterventionAmt2 = fit_params.iloc[5,n+1]
        InterventionAmt3 = fit_params.iloc[19,n+1]
        D_hospital_lag_GA = fit_params.iloc[6,n+1]
        D_hospital_lag_ICU = fit_params.iloc[7,n+1]
        D_recovery_GA = fit_params.iloc[8,n+1]
        D_recovery_ICU = fit_params.iloc[9,n+1]
        D_death_GA = fit_params.iloc[10,n+1]
        D_death_ICU = fit_params.iloc[11,n+1]
        P_SEVERE = fit_params.iloc[12,n+1]
        p_ICU = fit_params.iloc[13,n+1]
        CFR_GA = fit_params.iloc[14,n+1]
        CFR_ICU = fit_params.iloc[15,n+1]
        D_infectious_asym = fit_params.iloc[16,n+1]
        p_sym = fit_params.iloc[17,n+1]
        
        # Generate data using fitted parameters
        t_shift = t + t0
        xfit = SEIRfun.SEIR_results(t,population,
                                          InterventionTime,InterventionTime2,InterventionTime3,InterventionTime4,InterventionAmt,InterventionAmt2,InterventionAmt3,InterventionAmt4,R0,D_incubation,D_infectious,D_infectious_asym,
                                          P_SEVERE,CFR_GA,CFR_ICU,p_ICU,p_sym,D_recovery_asym,D_recovery_mild,D_hospital_lag_GA,D_hospital_lag_ICU,D_recovery_GA,D_recovery_ICU,D_death_GA,D_death_ICU)
        
        # interpolate to 't_final' so that all model results for all regions are eqivalent
        interp = interp1d(t_shift, xfit, kind='cubic', axis = 0, fill_value="extrapolate")
        x_interpolated = interp(t_final) 
        
        # Re-calculate daily figures after interpolation, in order to get daily changes
        x_interpolated[1:,discharge_daily_col_idx] = np.diff(x_interpolated[:,idx_R_hospitalised]) # rate of change in discharged patients, based on stepwise increase in recovered from hospital
        x_interpolated[0,discharge_daily_col_idx] = 0 # insert 0 as first value in daily column
        x_interpolated[1:,fatality_daily_col_idx] = np.diff(x_interpolated[:,idx_R_fatal]) # rate of change in fatalities, from stepwise increase in deaths
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
        stats[2] = max(SEIR_table_list[n][:,idx_hospitalised_ICU]) # size 
        stats[3] = t_final[np.where(SEIR_table_list[n][:,idx_hospitalised_ICU] == max(SEIR_table_list[n][:,idx_hospitalised_ICU]))] #  date
        # size and date of daily discharges peak
        stats[4] = max(SEIR_table_list[n][:,discharge_daily_col_idx]) # size 
        stats[5] = t_final[np.where(SEIR_table_list[n][:,discharge_daily_col_idx] == max(SEIR_table_list[n][:,discharge_daily_col_idx]))] #  date
        # size and date of daily deaths peak
        stats[6] = max(SEIR_table_list[n][:,fatality_daily_col_idx]) # size 
        stats[7] = t_final[np.where(SEIR_table_list[n][:,fatality_daily_col_idx] == max(SEIR_table_list[n][:,fatality_daily_col_idx]))] #  date
        
        stats[8] = max(SEIR_table_list[n][:,idx_R_hospitalised]) # cumulative number of discharges (recovered from severe)
        stats[9] = max(SEIR_table_list[n][:,idx_R_fatal]) # cumulative number of deaths
        stats[10] = stats[8] + stats[9] # cumulative number of hospitalised (recovered(severe (hospitalised)) + deaths)
          
        stats_list.append(stats)
        
        
        #%% Plot predicted against actuals
        pred_lines = 'dashed'
        
        plt.figure()
        #plt.plot(t_final-t0, SEIR_table_list[n][:,idx_cases], color='gold',linestyle=pred_lines) # total cases
        plt.plot(t_final-t0, SEIR_table_list[n][:,hosp_col_idx], '-b',linestyle=pred_lines) # hospitalised daily
        plt.plot(t_final-t0, SEIR_table_list[n][:,idx_hospitalised_ICU], '-r',linestyle=pred_lines) # ICU daily
        plt.plot(t_final-t0, SEIR_table_list[n][:,idx_R_hospitalised], '-g',linestyle=pred_lines) # recovered from severe
        plt.plot(t_final-t0, SEIR_table_list[n][:,idx_R_fatal], '-k',linestyle=pred_lines) # deaths
        if region_list[n] == 'NEL':
            #plt.plot(data_cases.date-t0, data_cases.iloc[:,1], color='gold', marker='o')
            plt.scatter(data_sitrep_hosp.Date-t0, data_sitrep_hosp.BHRUT_Barts_Hom, color='blue', marker='o')
            plt.scatter(data_sitrep_hosp.Date-t0, data_sitrep_hosp.BHRUT_Barts_Hom_ITU, color='red', marker='o')
            plt.scatter(data_sitrep_discharge.Date-t0, data_sitrep_discharge.CumBHRUTBartsHom, color='green', marker='o')
            plt.scatter(data_deaths.Date-t0, data_deaths.CumBHRUTBartsHom, color='black', marker='o')
            plt.title('NEL')
        elif region_list[n] == 'NEL trust sum':
            #plt.plot(data_cases.date-t0, data_cases.iloc[:,1], color='gold', marker='o')
            plt.scatter(data_sitrep_hosp.Date-t0, data_sitrep_hosp.BHRUT_Barts_Hom, color='blue', marker='o')
            plt.scatter(data_sitrep_hosp.Date-t0, data_sitrep_hosp.BHRUT_Barts_Hom_ITU, color='red', marker='o')
            plt.scatter(data_sitrep_discharge.Date-t0, data_sitrep_discharge.CumBHRUTBartsHom, color='green', marker='o')
            plt.scatter(data_deaths.Date-t0, data_deaths.CumBHRUTBartsHom, color='black', marker='o')
            plt.title('NEL (sum of trusts)')
        elif region_list[n] == 'BHRUT': 
            #plt.plot(data_cases.date-t0, data_cases.iloc[:,9], color='gold', marker='o')
            plt.scatter(data_sitrep_hosp.Date-t0, data_sitrep_hosp.BHRUT, color='blue', marker='o')
            plt.scatter(data_sitrep_hosp.Date-t0, data_sitrep_hosp.BHRUT_ITU, color='red', marker='o')
            plt.scatter(data_sitrep_discharge.Date-t0, data_sitrep_discharge.CumBHRUT, color='green', marker='o')
            plt.scatter(data_deaths.Date-t0, data_deaths.CumBHRUT, color='black', marker='o')
            plt.title('BHRUT')
        elif region_list[n] == 'Barts':
            #plt.plot(data_cases.date-t0, data_cases.iloc[:,10], color='gold', marker='o')
            plt.scatter(data_sitrep_hosp.Date-t0, data_sitrep_hosp.Barts, color='blue', marker='o')
            plt.scatter(data_sitrep_hosp.Date-t0, data_sitrep_hosp.Barts_ITU, color='red', marker='o')
            plt.scatter(data_sitrep_discharge.Date-t0, data_sitrep_discharge.CumBarts, color='green', marker='o')
            plt.scatter(data_deaths.Date-t0, data_deaths.CumBarts, color='black', marker='o')
            plt.title('Barts Health')
        elif region_list[n] == 'Homerton':
            #plt.plot(data_cases.date-t0, data_cases.iloc[:,3], color='gold', marker='o')
            plt.scatter(data_sitrep_hosp.Date-t0, data_sitrep_hosp.Homerton, color='blue', marker='o')
            plt.scatter(data_sitrep_hosp.Date-t0, data_sitrep_hosp.Homerton_ITU, color='red', marker='o')
            plt.scatter(data_sitrep_discharge.Date-t0, data_sitrep_discharge.CumHomerton, color='green', marker='o')
            plt.scatter(data_deaths.Date-t0, data_deaths.CumHomerton, color='black', marker='o')
            plt.title('Homerton UT')
            
        plt.xlabel('Day')
        plt.ylabel('Patients')
        plt.legend(['Hospitalised (predicted)', 'ITU (predicted)','Discharged (predicted)', 'Hospital deaths (predicted)', 'Hospitalised (actual, NHSE/I)','ITU (actual, NHSE/I)','Discharged (actual, NHSE/I)','Hospital deaths (actual, NHSE)'], loc='best')
        plt.show() 
        
        #%% Plot R(t)      
        plt.figure()
        plt.plot(t_final-t0, SEIR_table_list[n][:,idx_R], '-b') 
        plt.xlabel('Day')
        plt.ylabel('R(t)')
        plt.title('R vs. time')
        plt.show()         
        
        plt.figure()
        plt.plot_date(t_final_dateformat, SEIR_table_list[n][:,idx_R], '-b')
        plt.xlabel('Date')
        plt.ylabel('R(t)')
        plt.title('R vs. time')
        plt.gcf().autofmt_xdate()
        myFmt = mdates.DateFormatter('%m:%y')
        plt.gca().xaxis.set_major_formatter(myFmt)
        
    #%% Create borough level breakdowns by applying proportion of PHE confirmed cases by borough to NEL (sum of 3 trust fits)
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
        stats[2] = max(SEIR_table_list_borough[n][:,idx_hospitalised_ICU]) # size 
        stats[3] = t_final[np.where(SEIR_table_list_borough[n][:,idx_hospitalised_ICU] == max(SEIR_table_list_borough[n][:,idx_hospitalised_ICU]))] #  date
        # size and date of daily discharges peak
        stats[4] = max(SEIR_table_list_borough[n][:,discharge_daily_col_idx]) # size 
        stats[5] = t_final[np.where(SEIR_table_list_borough[n][:,discharge_daily_col_idx] == max(SEIR_table_list_borough[n][:,discharge_daily_col_idx]))] #  date
        # size and date of daily deaths peak
        stats[6] = max(SEIR_table_list_borough[n][:,fatality_daily_col_idx]) # size 
        stats[7] = t_final[np.where(SEIR_table_list_borough[n][:,fatality_daily_col_idx] == max(SEIR_table_list_borough[n][:,fatality_daily_col_idx]))] #  date
        
        stats[8] = max(SEIR_table_list_borough[n][:,idx_R_hospitalised]) # cumulative number of discharges (recovered from severe)
        stats[9] = max(SEIR_table_list_borough[n][:,idx_R_fatal]) # cumulative number of deaths
        stats[10] = stats[8] + stats[9] # cumulative number of hospitalised (recovered(severe (hospitalised)) + deaths)
        
        stats_list_borough.append(stats)
    


#%% Post-lockdown scenarios
# VALIDATE CAPACITIES AND ADD OTHER RELEVANT FIGURES
capacity_all = 3000
capacity_ICU = 800

if do_scenarios == 1: # 1 to do scenario mapping
    InterventionTime4_day_idx = np.where(t_final == InterventionTime4_day)[0][0]
    pop_idx = [0, 8, 9, 2] # index locations of relevant populations 
    
    # set range of options for fractional change in beta hat before intervention after Intervention 4
    dr = 0.05
    r_max = 1
    InterventionAmt4_range = np.linspace(0.0,r_max,int((r_max+dr)/dr))
    
    c_list = np.empty(InterventionAmt4_range.shape[0]) # calculate equivalent beta and contact rate, c
    scenario_hosp_peak = np.empty(InterventionAmt4_range.shape[0])
    scenario_hosp_peak_date = np.empty(InterventionAmt4_range.shape[0])
    scenario_ICU_peak = np.empty(InterventionAmt4_range.shape[0])
    scenario_ICU_peak_date = np.empty(InterventionAmt4_range.shape[0])
    
    data_min_dateformat = datetime.datetime(2020, 3, 6, 00, 00) # 43896 = 06/03/2020
    data_dateformat = pd.date_range(data_min_dateformat, periods=data_sitrep_hosp.Date.shape[0]).tolist()
    data_dates = matplotlib.dates.date2num(data_dateformat)
    
    plt.figure()
    plt.scatter(data_dates, data_sitrep_hosp.BHRUT_Barts_Hom, color='blue', marker='o')
    plt.scatter(data_dates, data_sitrep_hosp.BHRUT_Barts_Hom_ITU, color='red', marker='o')
#    plt.scatter(data_sitrep_hosp.Date, data_sitrep_hosp.BHRUT_Barts_Hom, color='blue', marker='o')
#    plt.scatter(data_sitrep_hosp.Date, data_sitrep_hosp.BHRUT_Barts_Hom_ITU, color='red', marker='o')
    
    for y in range(0,InterventionAmt4_range.shape[0],1):
        print(y)
        stats_list = []
        SEIR_table_list = []
        # Generate final SEIR tables for different regions
        for n in range(0,1,1):#fit_params.shape[1]-1,1): # for each region
            #print(n)
            population = data_pop.iloc[0,pop_idx[n]] #population
            
            # Define parameters obtained from fitting
            R0 = fit_params.iloc[2,n+1]
            t0 = fit_params.iloc[3,n+1]
            InterventionTime = InterventionTime1_day-t0
            InterventionTime2 = InterventionTime2_day-t0
            InterventionTime3 = InterventionTime3_day-t0
            InterventionAmt = fit_params.iloc[4,n+1]
            InterventionAmt2 = fit_params.iloc[5,n+1]
            InterventionAmt3 = fit_params.iloc[19,n+1]
            D_hospital_lag_GA = fit_params.iloc[6,n+1]
            D_hospital_lag_ICU = fit_params.iloc[7,n+1]
            D_recovery_GA = fit_params.iloc[8,n+1]
            D_recovery_ICU = fit_params.iloc[9,n+1]
            D_death_GA = fit_params.iloc[10,n+1]
            D_death_ICU = fit_params.iloc[11,n+1]
            P_SEVERE = fit_params.iloc[12,n+1]
            p_ICU = fit_params.iloc[13,n+1]
            CFR_GA = fit_params.iloc[14,n+1]
            CFR_ICU = fit_params.iloc[15,n+1]
            D_infectious_asym = fit_params.iloc[16,n+1]
            p_sym = fit_params.iloc[17,n+1]
            
            # Define Intervention 4 parameters
            InterventionTime4 = InterventionTime4_day-t0
            InterventionAmt4 = InterventionAmt4_range[y]
            
            # Generate data using fitted parameters
            t_shift = t + t0
            xfit = SEIRfun.SEIR_results(t,population,
                                              InterventionTime,InterventionTime2,InterventionTime3,InterventionTime4,InterventionAmt,InterventionAmt2,InterventionAmt3,InterventionAmt4,R0,D_incubation,D_infectious,D_infectious_asym,
                                              P_SEVERE,CFR_GA,CFR_ICU,p_ICU,p_sym,D_recovery_asym,D_recovery_mild,D_hospital_lag_GA,D_hospital_lag_ICU,D_recovery_GA,D_recovery_ICU,D_death_GA,D_death_ICU)
            
            # interpolate to 't_final' so that all model results for all regions are eqivalent
            interp = interp1d(t_shift, xfit, kind='cubic', axis = 0, fill_value="extrapolate")
            x_interpolated = interp(t_final) 
            
            # Re-calculate daily figures after interpolation, in order to get daily changes
            x_interpolated[1:,discharge_daily_col_idx] = np.diff(x_interpolated[:,idx_R_hospitalised]) # rate of change in discharged patients, based on stepwise increase in recovered from hospital
            x_interpolated[0,discharge_daily_col_idx] = 0 # insert 0 as first value in daily column
            x_interpolated[1:,fatality_daily_col_idx] = np.diff(x_interpolated[:,idx_R_fatal]) # rate of change in fatalities, from stepwise increase in deaths
            x_interpolated[0,fatality_daily_col_idx] = 0 # insert 0 as first value in daily column
            
            SEIR_table_list.append(x_interpolated) 
            
#            SEIR_table_list.append(SEIR_table_list[1] + SEIR_table_list[2] + SEIR_table_list[3]) # NEL sum of 3 trusts
        
        # Generate Summary Stats
        for n in range(0,len(SEIR_table_list),1): # for each region    
            # summary stats
            stats = np.zeros([11])
            # size and date of daily hospitalisation peak
            stats[0] = max(SEIR_table_list[n][InterventionTime4_day_idx:,hosp_col_idx]) # size 
            stats[1] = t_final[np.where(SEIR_table_list[n][:,hosp_col_idx] == max(SEIR_table_list[n][InterventionTime4_day_idx:,hosp_col_idx]))] #  date
            # size and date of daily ICU hospitalisation peak
            stats[2] = max(SEIR_table_list[n][InterventionTime4_day_idx:,idx_hospitalised_ICU]) # size 
            stats[3] = t_final[np.where(SEIR_table_list[n][:,idx_hospitalised_ICU] == max(SEIR_table_list[n][InterventionTime4_day_idx:,idx_hospitalised_ICU]))] #  date
            # size and date of daily discharges peak
            stats[4] = max(SEIR_table_list[n][InterventionTime4_day_idx:,discharge_daily_col_idx]) # size 
            stats[5] = t_final[np.where(SEIR_table_list[n][:,discharge_daily_col_idx] == max(SEIR_table_list[n][InterventionTime4_day_idx:,discharge_daily_col_idx]))] #  date
            # size and date of daily deaths peak
            stats[6] = max(SEIR_table_list[n][InterventionTime4_day_idx:,fatality_daily_col_idx]) # size 
            stats[7] = t_final[np.where(SEIR_table_list[n][:,fatality_daily_col_idx] == max(SEIR_table_list[n][InterventionTime4_day_idx:,fatality_daily_col_idx]))] #  date
            
            stats[8] = max(SEIR_table_list[n][:,idx_R_hospitalised]) # cumulative number of discharges (recovered from severe)
            stats[9] = max(SEIR_table_list[n][:,idx_R_fatal]) # cumulative number of deaths
            stats[10] = stats[8] + stats[9] # cumulative number of hospitalised (recovered(severe (hospitalised)) + deaths)
              
            stats_list.append(stats)    

        # peak hospitalisation for given intervention 3 and 4 effects
        scenario_hosp_peak[y] = stats_list[-1][0] # NEL sum hospitalisation peak
        scenario_hosp_peak_date[y] = stats_list[-1][1] # NEL sum hospitalisation peak date
        
        scenario_ICU_peak[y] = stats_list[-1][2] # NEL sum hospitalisation peak
        scenario_ICU_peak_date[y] = stats_list[-1][3] # NEL sum hospitalisation peak date

        c_list[y] = xfit[-1][-2] # get final c from the final time value

        #%% plot hospitalisation vs. time for various scenarios
        pred_lines = 'dashed'
        
#        plt.figure()
        plt.plot_date(t_final_dateformat, SEIR_table_list[n][:,hosp_col_idx], '-b',linestyle=pred_lines)
        plt.plot_date(t_final_dateformat, SEIR_table_list[n][:,idx_hospitalised_ICU], '-r',linestyle=pred_lines)
#        plt.plot(t_final_dateformat, SEIR_table_list[n][:,hosp_col_idx], '-b',linestyle=pred_lines) # hospitalised daily
#        plt.plot(t_final_dateformat, SEIR_table_list[n][:,idx_hospitalised_ICU], '-r',linestyle=pred_lines) # ICU daily
        plt.xlabel('Date')
        plt.ylabel('Hospitalisations')
        plt.legend(['Hospitalised (all)', 'Hospitalised (critical)'])
        plt.title('Hospitalisations for varying c')
        plt.gcf().autofmt_xdate()
        myFmt = mdates.DateFormatter('%m:%y')
        plt.gca().xaxis.set_major_formatter(myFmt)

    # vs. proportion of original beta
    plt.figure()
    plt.plot(InterventionAmt4_range, scenario_hosp_peak, '-b',linestyle=pred_lines,marker='o') # hospitalised peak
    plt.plot(InterventionAmt4_range, scenario_ICU_peak, '-r',linestyle=pred_lines,marker='o') # ICU peak
    plt.hlines(capacity_all, InterventionAmt4_range[0], InterventionAmt4_range[-1], colors='b', linestyles='dashed')
    plt.hlines(capacity_ICU, InterventionAmt4_range[0], InterventionAmt4_range[-1], colors='r', linestyles='dashed')
    plt.xlabel('proportion of pre-lockdown beta_hat')
    plt.ylabel('Peak hospitalisations (1/6/20 onwards == 2nd peak)')
    plt.legend(['Hospitalised (all)', 'Hospitalised (critical)','Capacity (all)','Capacity (critical)'])
    plt.title('2nd wave peak hospitalisations for varying proportion of pre-lockdown beta_hat')
    
    # vs. final c value
    plt.figure()
    plt.plot(c_list, scenario_hosp_peak, '-b',linestyle=pred_lines,marker='o') # hospitalised peak
    plt.plot(c_list, scenario_ICU_peak, '-r',linestyle=pred_lines,marker='o',) # ICU peak
    plt.hlines(capacity_all, c_list[0], c_list[-1], colors='b', linestyles='dashed')
    plt.hlines(capacity_ICU, c_list[0], c_list[-1], colors='r', linestyles='dashed')
    plt.xlabel('mean daily contacts, c')
    plt.ylabel('Peak hospitalisations (1/6/20 onwards == 2nd peak)')
    plt.legend(['Hospitalised (all)', 'Hospitalised (critical)','Capacity (all)','Capacity (critical)'])
    plt.title('2nd wave peak hospitalisations for varying c')
    plt.show() 