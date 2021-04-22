# 這個script後來沒用到
import sys
import csv
import statistics 

def inputCsvData(csvAddress):
    csvData = []
    with open(csvAddress, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            csvData.append(row)
    return csvData

def evaluate(videoType, istrain, evalCsvData):
    if(videoType == "ERP"):
        ClassroomVideoData = inputCsvData(f"code_silver/raw_datasets/states/ERP/{istrain}/ClassroomVideo.csv")
        TechnicolorHijackData = inputCsvData(f"code_silver/raw_datasets/states/ERP/{istrain}/TechnicolorHijack.csv")
        TechnicolorMuseumData = inputCsvData(f"code_silver/raw_datasets/states/ERP/{istrain}/TechnicolorMuseum.csv")
        allData = ClassroomVideoData + TechnicolorMuseumData + TechnicolorHijackData
        CEL = []
        for evalData in evalCsvData:
            for data in allData:
                if(evalData['Dataset'] == data['Dataset'] and evalData['Frame'] == data['Frame'] and 
                evalData['Synthesized.View'] == data['Synthesized.View'] and evalData['p1'] == data['p1'] and
                evalData['p2'] == data['p2'] and evalData['p3'] == data['p3']):
                    CEL.append(float(data['CEL']))
        print(f"avg.: {sum(CEL)/len(CEL)}")
        print(f"std.: {statistics.stdev(CEL)}")


    elif(videoType == "PTP"):
        pass


if __name__ == "__main__":
    evalCsvAddress = sys.argv[1]
    videoType = sys.argv[2]
    # only used in ERP video type
    istrain = sys.argv[3]

    evalCsvData = inputCsvData(evalCsvAddress)
    evaluate(videoType, istrain, evalCsvData)

