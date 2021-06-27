# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 16:42:40 2021

@author: bhavrathod
"""
# Imports
from getREefs import batchReEFs
from main import test_large, test_cplex
import pandas as pd
import numpy as np
import CoalPlants
import RenewableSites
from haversine import haversine, Unit
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt

# set some hardcoded values that will drive the analysis
def initialize():
    # Add states whose coal plants are to be analyzed (and possibly retired) as hardcoded values
    region = ['VA','MD']
    # Create file name using states we pulled data for OR use data points for all states in mid-atlantic
    # solFileName = 'solar_cf_'+'_'.join(region)+'_0.5_2014.csv'
    # winFileName = 'wind_cf_'+'_'.join(region)+'_0.5_2014.csv'
    # all states eligible for RE sites below
    solFileName = 'solar_cf_NY_PA_OH_WV_KY_TN_VA_MD_DE_NC_NJ_0.5_2014.csv'
    winFileName = 'wind_cf_NY_PA_OH_WV_KY_TN_VA_MD_DE_NC_NJ_0.5_2014.csv'
    # number of years the analysis will run for.
    numYears = 3
    # List of possible scenarios to test
    scen = {'ABG':[[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]}
    threshDist = 50
    return region, solFileName, winFileName, numYears, scen['ABG'][4], threshDist

# EFs are employment factors and determine how jobs are related with RE O&M and construction.
# Inputs: solar file name and wind file name to get all coordinates eligible for development, # of years for declination of EFs.
# Output: CONEF and REOMEF. These are arrays that are used by the model.
def getEFs(solFileName, winFileName, numYears):
    
    # Get construction EFs and RE O&M EFs for sites in csv files from cell above
#     CONEF,REOMEF = batchReEFs(solFileName,winFileName,numYears)
    # OR load the information from csv files saved from prior runs for above regions/numYears to save time.
    CONEF = np.loadtxt('CONEF_'+str(numYears)+'.csv', delimiter=',')
    REOMEF = np.loadtxt('REOMEF_'+str(numYears)+'.csv', delimiter=',')
    
    return CONEF, REOMEF

# Load coal plants in region and view their information.
# Input: region where coal plants are to be retired
# Output: DF with coal plants in analysis, locations, and metdata.
def loadCoalPlants(region):
    plants = CoalPlants.getCoalPlants(region)
    plants['HISTGEN'] = CoalPlants.getPlantGeneration(plants['Plant Code'])
    plants['HD'] = CoalPlants.getMarginalHealthCosts(plants['Plant Code'])
    plants.dropna(inplace=True)
    coalData = pd.read_excel('3_1_Generator_Y2019.xlsx',header=1,index_col='Plant Code',sheet_name=\
                             'Operable',usecols='B:F')
    coalPlants = plants.merge(coalData, left_on='Plant Code', right_index=True)
    coalPlants = coalPlants.drop_duplicates()
#     coalPlants.head()
    return coalPlants

# # of rows in reSites = num columns in solFileName and winFileName, in order.
# Output: a list of all RE sites that are eligible for RE investments.
def loadRESites(solFileName,winFileName):
    reSites = RenewableSites.getAnnualCF(solFileName,winFileName)
    reSitesL = list(reSites['Latitude'].astype(str)+','+reSites['Longitude'].astype(str)+','+reSites['Technology'].astype(str))
    return reSites, reSitesL

# Filter the RE sites to be within X miles from coal plants.
# Input: RE site coordinates, coal plant locations, and threshold distance from initialize
# Output: CSV file and MAXCAP numpy array which marks RE sites within threshold distance of a coal plant with 1000 MW capacity.
# SITEMAXCAP, which holds the maximum capacity of a RE site is then set to flat 1000 MW to limit the MW that can be built on a RE site that may be in vicinity of multiple coal plants.
def detMAXCAP(reSites,coalPlants, threshDist):
    MAXCAP = np.zeros((len(reSites),len(coalPlants)))
    SITEMAXCAP = np.zeros(len(reSites))
    
    reSites['Eligible'] = 0

    # for each coal plant, use its lat lon to calculate distance between RE sites and the plant. if distance is more than X then make capacity 0
    
    for c in range(MAXCAP.shape[1]):
        coalCord = (coalPlants.iloc[c,1],coalPlants.iloc[c,2])
        for s in range(MAXCAP.shape[0]):
            reCord = (reSites.iloc[s,0],reSites.iloc[s,1])
            dist = haversine(coalCord,reCord, unit=Unit.MILES)
            # if distance > threshold then set MAXCAP = 0. Else MAXCAP is 1000
            if dist<threshDist:
                MAXCAP[s,c] = 1000
                reSites.iloc[s,-1] = 1
    
    ind = reSites['Latitude'].astype(str)+reSites['Longitude'].astype(str)
    mCapDF = pd.DataFrame(MAXCAP,index=ind,columns=list(coalPlants['Plant Name']))
    
    mCapDF['S'] = mCapDF[list(mCapDF.columns)].sum(axis=1)
    mCapDF.to_csv('MAXCAP.csv')
    
    count = 0
    for s in mCapDF['S']:
        if s == 0:
            SITEMAXCAP[count]=0
        else:
            SITEMAXCAP[count]=1000
        count += 1
    
    return MAXCAP,reSites,SITEMAXCAP

# Solve the model. Model outputs obj object and ends with successful optimization confirmation.
def solveModel(scenario,numYears,solFileName,winFileName,region,CONEF,REOMEF,MAXCAP,SITEMAXCAP,reSites):
    a = scenario[0]
    b = scenario[1]
    g = scenario[2]
    obj, plants = test_cplex(a,b,g,numYears,solFileName,winFileName,region,CONEF,REOMEF,MAXCAP,SITEMAXCAP,reSites)
    return obj, plants
    
# What is the objective value returned?
def objValue(arg):
    print('Objective value is ${} million'.format(round(arg/10**6),2))

# Purpose: unpack the RE arrays in Obj product into a DF for easier processing. Save coal plant information with retirement year.
# Unpack results: for each year, for each coal, for each RE site do following:
def unpackResults(numYears,coalPlants,reSites,obj):
    
    cLat = []
    cLon = []
    pNam = []
    coalRetire = []
    coalOnline = []
    capRetire = []
    coalGen = []
    coalYr = []
    
    reOnline = []
    reInvest = []
    cpInvest = []
    totReCap = []
    renGenrn = []
    yr = []
    cPlant = []
    Lat = []
    Lon = []
    Typ = []
    CF = []
    elg = []

    # RE investment Lat/Lon/Type
    for y in range(numYears):
        cYr = y+2020
        
        for c in range(coalPlants.shape[0]):
            cLat.append(coalPlants.iloc[c,1])
            cLon.append(coalPlants.iloc[c,2])
            pNam.append(coalPlants.iloc[c,7])
            coalRetire.append(obj.coalRetire[c,y])
            coalOnline.append(obj.coalOnline[c,y])
            capRetire.append(obj.capRetire[c,y])
            coalGen.append(obj.coalGen[c,y])
            coalYr.append(cYr)
            
            for s in range(reSites.shape[0]):
                # If reOnline flag is set for site s for plant c and year y then add flags
                if obj.reOnline[s,c,y]==1:
                    reOnline.append(1)
                else:
                    reOnline.append(0)
                # If reInvest flag is set for site s for plant c and year y then add flags
                if obj.reInvest[s,c,y]==1:
                    reInvest.append(1)
                else:
                    reInvest.append(0)
                # If reInvest flag is set for site s for plant c and year y then add flags
                if obj.capInvest[s,c,y]>0:
                    cpInvest.append(obj.capInvest[s,c,y])
                else:
                    cpInvest.append(0)
                # If reInvest flag is set for site s for plant c and year y then add flags
                if obj.reCap[s,c,y]>0:
                    totReCap.append(obj.reCap[s,c,y])
                else:
                    totReCap.append(0)
                # If reInvest flag is set for site s for plant c and year y then add flags
                if obj.reGen[s,c,y]>0:
                    renGenrn.append(obj.reGen[s,c,y])
                else:
                    renGenrn.append(0)
                yr.append(cYr)
                cPlant.append(coalPlants.iloc[c,7])
                Lat.append(reSites.iloc[s,0])
                Lon.append(reSites.iloc[s,1])
                Typ.append(reSites.iloc[s,3])
                CF.append(reSites.iloc[s,2])
                elg.append(reSites.iloc[s,-1])
    
    # Create coal data CSV file.
    dat = {'Year':coalYr,'Lat':cLat,'Lon':cLon,'coalOnline':coalOnline,'coalGen':coalGen,'coalRetire':coalRetire,\
           'capRetire':capRetire}
    coalData = pd.DataFrame(dat)
    coalData.to_csv('coalData.csv')
    
    dat = {'Year':yr,'Lat':Lat,'Lon':Lon,'Type':Typ,'Ann.CF':CF,'EligibleSite':elg,'Online':reOnline,'Investment':reInvest,\
           'Invested MW':cpInvest,'Total MW Cap.':totReCap,'Tot MWh Gen':renGenrn,'Repl. Plant':cPlant}
    reData = pd.DataFrame(dat)
    reData.to_csv('reData.csv')
    return coalData,reData
    
# plot coal plant retirements
def coalPlantRet(numYears,arr):
    xYears = np.linspace(2020,2020+numYears-1,numYears)
    coalRet_Y = []
    for y in range(numYears):
        coalRet_Y.append(sum(arr[:,y]))
    plt.scatter(xYears,coalRet_Y,label='Coal ret. ind.')
    plt.xlabel('Years->')
    plt.ylabel('# of occurences')
    plt.grid(True,linestyle='-')
    return plt,xYears

# plot online coal plants
def coalPlantOnline(xYears,arr,plt):
    coalOnl_Y = []
    for y in range(len(xYears)):
        coalOnl_Y.append(sum(arr[:,y]))
    plt.scatter(xYears,coalOnl_Y,label='Coal online ind.')
    plt.grid(True,linestyle='-')
    return plt

# plot coal capacity retirement information
def coalCapRet(xYears,arr,coalPlants):
    capRet_Y = []
    for y in range(len(xYears)):
        capRet_Y.append(sum(arr[:,y]))
    plt.bar(xYears,capRet_Y,label='Retired coal capacity')
    plt.xlabel('Years->')
    plt.ylabel('Capacity [MW]')
    plt.axhline(y = round(coalPlants['Coal Capacity (MW)'].sum()),ls='--',c='r')
    plt.grid(True,linestyle='-')
    return plt

# plot coal capacity retirement information
def coalGen(xYears,arr):
    coalGen_Y = []
    for y in range(len(xYears)):
        coalGen_Y.append(sum(arr[:,y]))
    plt.bar(xYears,coalGen_Y,label='Coal generation')
    plt.xlabel('Years->')
    plt.ylabel('Generation [MWh]')
    plt.grid(True,linestyle='--')
    return plt

# plot coal capacity retirement information
def reIndicators(xYears,arr1,arr2,coalPlants,numYears):
    reInvest = []
    reOnline = []
    for y in range(numYears):
        inv = 0
        onl = 0
        for c in range(coalPlants.shape[0]):
            onl += sum(arr1[:,c,y])
            inv += sum(arr2[:,c,y])
        reInvest.append(inv)
        reOnline.append(onl)
    plt.scatter(xYears,reOnline,label='# RE Online')
    plt.scatter(xYears,reInvest,label='RE Investment',marker='x')
    plt.xlabel('Years->')
    plt.ylabel('Count')
    plt.grid(True,linestyle='--')
    return plt

# plot RE capacity investment and total RE capacity over time.
def reCapInvest(xYears,arr1,arr2,coalPlants,numYears):
    capInvest = []
    reCapT = []
    for y in range(numYears):
        inv = 0
        cap = 0
        for c in range(coalPlants.shape[0]):
            inv += sum(arr1[:,c,y])
            cap += sum(arr2[:,c,y])
        capInvest.append(inv)
        reCapT.append(cap)
    plt.scatter(xYears,capInvest,label='New capacity')
    plt.scatter(xYears,reCapT,label='Total capacity',marker='x')
    plt.xlabel('Years->')
    plt.ylabel('RE MW investments and totals')
    plt.grid(True,linestyle='--')
    return plt

# plot RE generation vs total HISTGEN lost over time.
def genCompare(xYears,arr1,coalRet,coalPlants,numYears):
    reGen = []
    hGen = []
    for y in range(numYears):
        lossGen = 0
        reGenMWh = 0
        for c in range(coalPlants.shape[0]):
            if coalRet[c,y] ==1:
                lossGen += coalPlants.iloc[c,4]
                coalPlants.iloc[c,10]=y+2020
            reGenMWh += sum(arr1[:,c,y])
        reGen.append(reGenMWh)
        hGen.append(lossGen)
    plt.scatter(xYears,reGen,label='RE Generation')
    plt.bar(xYears,hGen,label='Lost HISTGEN')
    plt.xlabel('Years->')
    plt.ylabel('Generation MWh')
    plt.grid(True,linestyle='--')
    return plt

# Purpose: call multiple plotting functions for quick visualization of information.
def createPlots(numYears,obj, coalPlants):
    plt,xYears = coalPlantRet(numYears,obj.coalRetire)
    plt = coalPlantOnline(xYears,obj.coalOnline,plt)
    plt.legend()
    plt.show()
    plt = coalCapRet(xYears,obj.capRetire,coalPlants)
    plt.legend()
    plt.show()
    plt = coalGen(xYears,obj.coalGen)
    plt.legend()
    plt.show()
    plt = reIndicators(xYears,obj.reOnline,obj.reInvest,coalPlants,numYears)
    plt.legend()
    plt.show()
    plt = reCapInvest(xYears,obj.capInvest,obj.reCap,coalPlants,numYears)
    plt.legend()
    plt.show()
    coalPlants['retYear'] = 0
    plt = genCompare(xYears,obj.reGen,obj.coalRetire,coalPlants,numYears)
    plt.legend()
    plt.show()
    coalPlants.to_csv('coalRetInfo.csv')

# Create a Folium Map with multiple Layers    
def vizFolium(reData,coalPlants):
    
    # centered on Onion Maiden restaurant in Pittsburgh PA
    m = folium.Map(
        location=[40.42185334811013, -79.99594457857727],
        tiles="Cartodb positron",
        zoom_start=4
    )
    def detCol(arg):
        if arg=='s':
            return 'orange'
        elif arg=='w':
            return 'blue'
        else:
            return 'green'
    
    # container for coal plant locations.
    coalFG = folium.FeatureGroup(name='Coal plant locations')
    df = coalPlants
    for c in range(df.shape[0]):
        popText = str(df.iloc[c,1])+str(df.iloc[c,2])+', '+str(df.iloc[c,7])+', '+str(df.iloc[c,3])+' MW, HD '+str(round(df.iloc[c,5],2))
        folium.Circle(
            location=[df.iloc[c,1],df.iloc[c,2]],
            tooltip=popText,
            popup=popText,
            radius=3.0,
            color='red'
        ).add_to(coalFG)
    coalFG.add_to(m)
    
    # show all eligible coal plant locations.
    elSitesFG = folium.FeatureGroup(name='Eligible RE locations', show=False)
    df = reData.loc[(reData['EligibleSite']==1) & (reData['Year']==2020)]
    for c in range(df.shape[0]):
        popText = str(df.iloc[c,1])+str(df.iloc[c,2])+', '+str(df.iloc[c,7])
        folium.CircleMarker(
            location=[df.iloc[c,1],df.iloc[c,2]],
            tooltip=popText,
            popup=popText,
            weight=0.5,
            color='grey'
        ).add_to(elSitesFG)
    elSitesFG.add_to(m)
    
    # locate where RE Investments happened in year 2020
    reInvestFG = folium.FeatureGroup(name='Sites w/ Investments',show=False)
    marker_cluster = MarkerCluster().add_to(reInvestFG)
    df = reData.loc[(reData['Investment']==1) & (reData['Year']==2020)]
    for c in range(df.shape[0]):
        popText = str(df.iloc[c,1])+str(df.iloc[c,2])+', Type:'+str(df.iloc[c,3])\
        +', '+str(df.iloc[c,8])+' MW'
        folium.CircleMarker(
            location=[df.iloc[c,1],df.iloc[c,2]],
            tooltip=popText,
            popup=popText,
            color=detCol(df.iloc[c,3]),
            radius=4,
        ).add_to(marker_cluster)
    reInvestFG.add_to(m)
    
    # locate online RE plants in year 2020
    onlineFG = folium.FeatureGroup(name='Sites online 2020',show=False)
    marker_cluster = MarkerCluster().add_to(onlineFG)
    df = reData.loc[(reData['Online']==1) & (reData['Year']==2020)]
    for c in range(df.shape[0]):
        popText = str(df.iloc[c,1])+str(df.iloc[c,2])+', Type:'+str(df.iloc[c,3])\
        +', '+str(df.iloc[c,8])+' MW'
        folium.Circle(
            location=[df.iloc[c,1],df.iloc[c,2]],
            tooltip=popText,
            popup=popText,
            color='green',
            radius=1.5,
        ).add_to(marker_cluster)
    onlineFG.add_to(m)
    
    # locate where RE Investments happened in year 2020
    validInvFG = folium.FeatureGroup(name='0+ MW RE capacity',show=False)
    marker_cluster = MarkerCluster().add_to(validInvFG)
    df = reData.loc[(reData['Investment']==1) & (reData['Year']==2020) & (reData['Invested MW']!=0)]
    for c in range(df.shape[0]):
        popText = str(df.iloc[c,1])+str(df.iloc[c,2])+', Type:'+str(df.iloc[c,3])\
        +', '+str(df.iloc[c,8])+' MW'
        folium.Marker(
            location=[df.iloc[c,1],df.iloc[c,2]],
            tooltip=popText,
            popup=popText,
            icon=folium.Icon(color=detCol(df.iloc[c,3]))
        ).add_to(marker_cluster)
    validInvFG.add_to(m)
    
    def detRad(arg):
        if arg<50:
            return 5
        elif arg<5000:
            return 10
        elif arg<1200000:
            return 15
        else:
            return 20

    # locate where RE Investments happened in year 2020
    reGenFG = folium.FeatureGroup(name='RE Gen Magnitude',show=False)
    df['Loc']=df['Lat'].astype(str)+df['Lon'].astype(str)
    totGen = []
    totMW = []
    types = []
    Lat = []
    Lon = []
    for l in df['Loc'].unique():
        tDF = df.loc[df['Loc']==l]
        types.append('_'.join(list(set(tDF.Type.values))))
        totGen.append(tDF['Tot MWh Gen'].sum())
        totMW.append(tDF['Total MW Cap.'].sum())
        Lat.append(tDF.iloc[0,1])
        Lon.append(tDF.iloc[0,2])
    d = {'Lt':Lat,'Ln':Lon,'totGen':totGen,'totMW':totMW,'types':types}
    df1 = pd.DataFrame(d)
    for c in range(df1.shape[0]):
        popText = str(df1.iloc[c,0])+str(df1.iloc[c,1])+', Type:'+str(df1.iloc[c,4])\
        +', '+str(df1.iloc[c,2])+' MWh'
        folium.CircleMarker(
            location=[df1.iloc[c,0],df1.iloc[c,1]],
            tooltip=popText,
            popup=popText,
            radius = detRad(df1.iloc[c,2]),
        ).add_to(reGenFG)
    reGenFG.add_to(m)
    folium.LayerControl().add_to(m)
    return m

# Purpose: check optimization formulation constraints 
def checkConstraints(numYears,obj,coalPlants,SITEMAXCAP,MAXCAP):
    # coalGenRule: coal generation must equal HISTGEN if a plant is online. Change year index below to check other years.
    for y in range(numYears):
        print(2020+y, obj.coalGen[:,y]==coalPlants.HISTGEN.values*obj.coalOnline[:,y])
        
    # balanceGenRule: RE generation replacing the coal plants must equal retired capacity (i.e. difference between HISTGEN and coalGen)
    for y in range(numYears):
        print(2020+y,sum(sum(obj.reGen[:,:,0]))==sum(coalPlants.HISTGEN.values)-sum(obj.coalGen[:,0]))
    
    # reCapRule: RE capacity at a site is <= MAXCAPACITY * if the plant is online
    for c in range(coalPlants.shape[0]):
        for y in range(numYears):
            # Since we dont know which exact sites are online, sum all sites for each plant for each year and compare that way.
            # Since indices on reCap, MAXCAP and reOnline correspond to same sites, this validation works
            print(c,2020+y,sum(obj.reCap[:,c,y])<=sum(MAXCAP[:,c])*sum(obj.reOnline[:,c,y]))
            
    # reCapLimit: RE plants have a maximum site capacity of 1000 MW.
    # Since we dont know the indices of eligible site, validate that sum of all RE Cap <= sum of all SITEMAXCAP
    for y in range(numYears):
        print(2020+y,sum(sum(obj.reCap[:,:,0]))<=sum(SITEMAXCAP))
        
    # capInvestRule: Invested MW capacity = reCap_y - reCap_y-1
    print(sum(sum(obj.capInvest[:,:,0]))==sum(sum(obj.reCap[:,:,0])))
    print(sum(sum(obj.capInvest[:,:,1]))==sum(sum(obj.reCap[:,:,1]))-sum(sum(obj.reCap[:,:,0])))
    
def countReInfo(SITEMAXCAP,reData):
    print('Total {} sites can have RE capacity (\'valid\').'.format(int(SITEMAXCAP.sum()/1000)))
    print('\n*RE Online Indicator*')
    print('{} RE plants are online in 2020'.format(reData.loc[(reData['Online']==1) & (reData['Year']==2020)].shape[0]))
    print('\t{} plants are ineligible (beyond the threshold distance)'.format(reData.loc[(reData['Online']==1) & (reData['Year']==2020) & (reData['EligibleSite']==0)].shape[0]))
    print('\t{} plants are in eligible locations.'.format(reData.loc[(reData['Online']==1) & (reData['Year']==2020)].shape[0]-reData.loc[(reData['Online']==1) & (reData['Year']==2020) & (reData['EligibleSite']==0)].shape[0]))
    print('\n*RE Investment Indicator*')
    print('{} RE sites with investments in 2020'.format(reData.loc[(reData['Investment']==1) & (reData['Year']==2020)].shape[0]))
    print('\t{} sites are ineligible (beyond the threshold distance)'.format(reData.loc[(reData['Investment']==1) & (reData['Year']==2020) & (reData['EligibleSite']==0)].shape[0]))
    print('\t{} sites are in eligible locations.'.format(reData.loc[(reData['Investment']==1) & (reData['Year']==2020)].shape[0]-reData.loc[(reData['Investment']==1) & (reData['Year']==2020) & (reData['EligibleSite']==0)].shape[0]))
    print('\n*Valid RE Investment Locations*')
    print('{} RE sites with VALID investments in 2020'.format(reData.loc[(reData['Investment']==1) & (reData['Year']==2020) & (reData['Invested MW']!=0)].shape[0]))
    print('\t{} sites are ineligible (beyond the threshold distance)'.format(reData.loc[(reData['Investment']==1) & (reData['Year']==2020) & (reData['Invested MW']!=0) & (reData['EligibleSite']==0)].shape[0]))
    print('\t{} sites are in eligible locations.'.format(reData.loc[(reData['Investment']==1) & (reData['Year']==2020) & (reData['Invested MW']!=0)].shape[0]-reData.loc[(reData['Investment']==1) & (reData['Invested MW']!=0) & (reData['Year']==2020) & (reData['EligibleSite']==0)].shape[0]))

def main():
    region, solFileName, winFileName, numYears, scenario, threshDist = initialize()
    CONEF, REOMEF = getEFs(solFileName, winFileName, numYears)
    coalPlants = loadCoalPlants(region)
    reSites, reSitesL = loadRESites(solFileName,winFileName)
    MAXCAP,reSites,SITEMAXCAP = detMAXCAP(reSites,coalPlants,threshDist)
    # use below to reset index and prevent Pyomo warnings.
    reSites = reSites.reset_index(drop=True)
    obj, plants = solveModel(scenario,numYears,solFileName,winFileName,region,CONEF,REOMEF,MAXCAP,SITEMAXCAP,reSites)
    objValue(obj.Z)
    checkConstraints(numYears,obj,coalPlants,SITEMAXCAP,MAXCAP)
    createPlots(numYears,obj, coalPlants)
    coalData,reData = unpackResults(numYears,coalPlants,reSites,obj)
    print('See coalData DF for coal plant retirement information.')
    countReInfo(SITEMAXCAP,reData)
    m = vizFolium(reData,coalPlants)
    m.save('_'.join(region)+'_results.html')

main()