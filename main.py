import requests
import csv
from bs4 import BeautifulSoup

def num(s):
    # Convert to number or float depending on value
    try:
        return int(s)
    except ValueError:
        return float(s)

def getStatsForYear(years):
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

def writePlayerFile(data):

    with open('players.csv', mode='w') as player_file:
        player_writer = csv.writer(player_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # Write headers needed for csv use in weka
        player_writer.writerow(
            ['Comp', 'Att', 'Pct', 'Att/G', 'Yds', 'Avg', 'Yds/G', 'TD', 'Int', '1st', '1st%', 'Lng', '20+', '40+',
             'Sck', 'Rate'])

        # Write all player data to file
        for row in data:
            player_writer.writerow(row)

def classify():
    # Get current years incomplete data
    data = getStatsForYear([2018])
    # Set data to vars  so it  can be classified
    for row in data:
        Comp=row[0]
        Att=row[1]
        Pct=row[2]
        AttG=row[3]
        Yds=row[4]
        Avg=row[5]
        YdsG=row[6]
        TD=row[7]
        Int=row[8]
        oneSt=row[9]
        onestPer=row[10]
        Lng=row[11]
        twoPlus=row[12]
        fourPlus=row[13]
        Sck=row[14]
        Rate=row[15]

        # This is where the magic happens
        RateNew = -0.0581 * Comp + 0.1195 * Att + 0.9745 * Pct + 0.4078 * AttG + -0.0036 * Yds + 7.0455 * Avg + -0.072 * YdsG + 0.8261 * TD + -1.2793 * Int + -0.1613 * oneSt + 0.6321 * onestPer + 0.0382 * Lng + 0.0344 * twoPlus + -48.3837

        # Print the actual rating and the one that's been classified
        print('NFL Rating: {0}, My Rating {1}'.format(Rate, round(RateNew, 2)))

stats = getStatsForYear(range(2017,1999,-1))

writePlayerFile(stats)
classify()
#print(stats)



