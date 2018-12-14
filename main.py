import requests
import csv
from bs4 import BeautifulSoup
import weka.core.jvm as jvm
from weka.classifiers import Classifier
from weka.core.converters import Loader, Saver
import matplotlib.pyplot as plt

def num(s):
    # Convert to number or float depending on value
    try:
        return int(s)
    except ValueError:
        return float(s)

"""
This functions gets all the stats for the NFl for the entered years and returns them

Parameters
----------
years : array
    The years from and to for the stats to be generated

Returns
-------
array
    A list of all player stats sorted into there own array per player

Raises
------
TODO: Sort This
KeyError
    when a key error
OtherError
    when an other error
"""
def getStatsForYearNFL(years):
    # Init Vars
    newData = []

    # For the supplied years
    for year in years:
        # Incase the second page needs to be evaluated(Data is irrelevant at the moment)
        for page in [1]:
            # Get player data
            data = requests.get(
                'http://www.nfl.com/stats/categorystats?archive=true&conference=null&statisticPositionCategory=QUARTERBACK&season='+
                str(year)+'&seasonType=REG&experience=&tabSeq=1&d-447263-p='+
                str(page)+'&qualified=true&Submit=Go')

            #Parse HTML into soup
            soup = BeautifulSoup(data.text, 'html.parser')
            # Find elements we care about
            for tr in soup.select('tbody > tr'):
                tempArr = []
                # Data we care about begins at 5
                for i in range(5, 21):
                    for td in tr.select('td:nth-of-type(' + str(i) + ')'):
                        # Sanitise the data to avoid problems down the line
                        temp = td.text.replace('\n', '')
                        temp = temp.replace('\t', '')
                        temp = temp.replace(',', '')
                        temp = temp.replace('T', '')
                        temp = temp.replace('--', '0')
                        # Add player data element to arr
                        tempArr.append(num(temp))
                # Add player data row to main arr
                newData.append(tempArr)
    return newData

def getStatsForYearCOL(years):
    # Init Vars
    newData = []

    # For the supplied years
    for year in years:
        # Incase the second page needs to be evaluated(Data is irrelevant at the moment)
        for page in [1]:
            # Get player data
            if page==1:
                page =''
            else:
                page=str((page*40)+1)
            data = requests.get('http://www.espn.com/college-football/statistics/player/_/stat/passing/sort/passingYards/year/'+str(year)+'/qualified/false'+page)

            #Parse HTML into soup
            soup = BeautifulSoup(data.text, 'html.parser')
            #print(soup.select('tr'))
            for bad in soup.findAll("tr", {'class': 'colhead'}):
                bad.decompose()
            # Find elements we care about
            for tr in soup.findAll('tr'):
                tempArr = []
                # Data we care about begins at 5
                for i in range(4, 13):
                    for td in tr.select('td:nth-of-type(' + str(i) + ')'):
                        # Sanitise the data to avoid problems down the line

                        # temp = td.text.replace('\n', '')
                        # temp = temp.replace('\t', '')
                        # temp = temp.replace(',', '')
                        # temp = temp.replace('T', '')
                        # temp = temp.replace('--', '0')
                        # Add player data element to arr
                        tempArr.append(num(td.text))

                    print(tempArr)
                # Add player data row to main arr
                newData.append(tempArr)
    return newData

"""
This functions writs data to an entered file name in the sam folder as the python file

Parameters
----------
fileName : array
    The name of the file the data should be stored in (should always end in .csv)
data : array
    The array of data to be contained within the file

Returns
-------
Null

Raises
------
TODO: Sort This
KeyError
    when a key error
OtherError
    when an other error
"""
def writePlayerFile(fileName,data):

    with open(fileName, mode='w') as player_file:
        player_writer = csv.writer(player_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # Write headers needed for csv use in weka
        player_writer.writerow(
            ['Comp', 'Att', 'Pct', 'Att/G', 'Yds', 'Avg', 'Yds/G', 'TD', 'Int', '1st', '1st%', 'Lng', '20+', '40+',
             'Sck', 'Rate'])

        # Write all player data to file
        for row in data:
            player_writer.writerow(row)

def classify(fileToClassify, fileToCompare, predictionYear=None,pastResultYears=None):
    # Start Java VM
    jvm.start(max_heap_size="1024m")
    # Load CSV files into weka loader
    loader = Loader(classname="weka.core.converters.CSVLoader")
    fileToClassifyData = loader.load_file(fileToClassify)
    fileToClassifyData.class_is_last()
    fileToCompareData = loader.load_file(fileToCompare)
    fileToCompareData.class_is_last()

    # Generate Classifier based on data
    classifier = Classifier(classname="weka.classifiers.functions.LinearRegression",
                            options=["-S", "0", "-R", "1.0E-8", "-num-decimal-places", "4"])
    classifier.build_classifier(fileToClassifyData)

    # Var builder for graph
    count = 0.0
    countPred = 0.0
    graphDetails = [['TITLE'],['NFL Data Ratings (Official) {0}'.format(pastResultYears), [], []], ['NFL Data Ratings (Predicted) {0}'.format(predictionYear), [], []]]

    # Time to predict results based on classifier
    for index, inst in enumerate(fileToCompareData):
        pred = classifier.classify_instance(inst)
        countPred+=pred
        count+=list(enumerate(inst))[15][1]
        print("{0:.3f} accurate compared to results.".format(countPred/count))

        dist = classifier.distribution_for_instance(inst)
        # NFL Results
        graphDetails[1][1].append(index + 1)
        graphDetails[1][2].append(list(enumerate(inst))[15][1])

        # Predicted Results
        graphDetails[2][1].append(index + 1)
        graphDetails[2][2].append(pred)
        print(str(index + 1) +": label index=" + str(pred) + ", class distribution=" + str(dist)+" , original: "+ str(list(enumerate(inst))[15][1]))
    graphDetails[0][0]='Player Rating Predictions For {0} ({1:.3f} Accurate)'.format(predictionYear,100-(countPred/count))
    jvm.stop()
    print(graphDetails)
    BuildGraph(graphDetails)

def BuildGraph(input):
    # Set Labels
    plt.xlabel('Player  Rank')
    plt.ylabel('Player Rating')

    plt.title(input.pop(0)[0])
    # Plot Data on  graph
    for row in input:
        plt.plot(row[1], row[2],label=row[0])

    # Add legend for data
    plt.legend()

    # Save and show fig
    plt.savefig('test.png')
    plt.show()

# Set Vars
# predictionYear =[2017]
# pastResultYears =[2016,1999]
# # Get stats for past years to classify
# statsPast = getStatsForYearNFL(range(pastResultYears[0],pastResultYears[1],-1))
# writePlayerFile('players_past.csv',statsPast)
# # Get last full stats for predictions
# stats = getStatsForYearNFL(predictionYear)
# writePlayerFile('players_predict.csv',stats)
#
# classify('players_past.csv','players_predict.csv',predictionYear="".join(map(str, predictionYear)),pastResultYears="-".join(map(str, pastResultYears)))

stats = getStatsForYearCOL([2017,2016])

# TODO: Have different functions for different graphs
# TODO: Have different functions for classifiers
