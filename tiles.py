import math
import numpy as np

tilesMaxX = 250
tilesMinX = -250
tilesMaxY = 250
tilesMinY = -250
tileSize = 5
nColumns = int(math.ceil((tilesMaxX-tilesMinX)/(tileSize)))
nRows = int(math.ceil((tilesMaxY-tilesMinY)/(tileSize)))

def getXBin(x, maxX = tilesMaxX, minX = tilesMinX, columns = nColumns):
    xRange = maxX - minX
    rX = float(columns)/(maxX - minX)
    xBin = int((x - minX) * rX)
    xBin = min(xBin, columns - 1)
    xBin = max(xBin, 0)
    return xBin

def getYBin(y, maxY = tilesMaxY, minY = tilesMinY, rows = nRows):
    yRange = maxY - minY
    rY = float(rows)/(maxY - minY)
    yBin = int((y - minY) * rY)
    yBin = min(yBin, rows - 1)
    yBin = max(yBin, 0)
    return yBin

def getGlobalBin(x, y, maxX = tilesMaxX, minX = tilesMinX, columns = nColumns, maxY = tilesMaxY, minY = tilesMinY, rows = nRows):
    return getXBin(x, maxX, minX, columns) + getYBin(y, maxY, minY, rows) * columns

def getGlobalBinByBin(xBin, yBin, columns = nColumns):
    return xBin + yBin * columns
  
def searchBox(xMin, xMax, yMin, yMax, maxX = tilesMaxX, minX = tilesMinX, columns = nColumns, maxY = tilesMaxY, minY = tilesMinY, rows = nRows):
    xBinMin = getXBin(xMin, maxX, minX, columns)
    xBinMax = getXBin(xMax, maxX, minX, columns)
    yBinMin = getYBin(yMin, maxY, minY, rows)
    yBinMax = getYBin(yMax, maxY, minY, rows)
    return [xBinMin, xBinMax, yBinMin, yBinMax]


if __name__ == "__main__":
    x = -243
    y = -243

    xBin = getXBin(x)
    yBin = getYBin(y)
    globalBin = getGlobalBin(x,y)
    globalBinByBin = getGlobalBinByBin(xBin, yBin)
    print("Columns: ", nColumns)
    print("Rows: ", nRows)

    print(xBin, yBin, globalBin, globalBinByBin)
    print("\nOpening a search box now")

    sbox = searchBox(x-6, x+6, y-10, y+10) 
    print(sbox)   