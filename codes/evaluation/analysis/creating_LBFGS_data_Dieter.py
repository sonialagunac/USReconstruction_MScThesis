"""
Sonia Laguna - ETH Zurich M.Sc. Thesis
Code to save the results on clinical VS data from Dieter Schweize and analyse them
"""
import numpy as np

# List of lesions studied
# FA: subjects = ['22_L1', '20_L2', '20_L1', '15_L1', '15_L2', '12_L2', '12_L1', '10_L1', '09_L1', '03_L2', '03_L1', '02_L1']
# CA: subjects = ['91_L1','45_L1', '42_L1', '21_L1', '19_L3', '19_L2', '19_L1', '17_L1','14_L1', '11_L1', '08_L1', '05_L1']

#Contrast based on "Selected inclusion value" SoS
dSoS_FA = []
dSoS_CA = []
dSoS_FA.append([np.abs(1496-1499),np.abs(1502-1503),np.abs(1502-1504),np.abs(1500-1515)])
dSoS_FA.append([np.abs(1485-1491),np.abs(1507-1511),np.abs(1505-1515),np.abs(1483-1490)])
dSoS_FA.append([np.abs(1480-1488),np.abs(1483-1493),np.abs(1485-1492),np.abs(1485-1492)])
dSoS_FA.append([np.abs(1518-1521),np.abs(1558-1562),np.abs(1534-1533),np.abs(1531-1536)])
dSoS_FA.append([np.abs(1512-1517),np.abs(1532-1539),np.abs(1532-1539),np.abs(1513-1517)])
dSoS_FA.append([np.abs(1519-1535),np.abs(1526-1531),np.abs(1517-1525),np.abs(1530-1533)])
dSoS_FA.append([np.abs(1534-1547),np.abs(1529-1533),np.abs(1514-1517),np.abs(1523-1532)])
dSoS_FA.append([np.abs(1539-1544),np.abs(1529-1535),np.abs(1515-1523),np.abs(1521-1528)])
dSoS_FA.append([np.abs(1520-1526),np.abs(1528-1537),np.abs(1523-1530),np.abs(1539-1544)])
dSoS_FA.append([np.abs(1537-1541),np.abs(1505-1507),np.abs(1533-1542),np.abs(1525-1530)])
dSoS_FA.append([np.abs(1508-1504),np.abs(1505-1508),np.abs(1531-1526),np.abs(1541-1547)])
dSoS_FA.append([np.abs(1546-1551),np.abs(1541-1547),np.abs(1543-1552),np.abs(1534-1543)])

dSoS_CA.append([np.abs(1470-1492),np.abs(1464-1487), np.abs(1466-1480),np.abs(1466-1479)])
dSoS_CA.append([np.abs(1476-1481),np.abs(1488-1492),np.abs(1491-1497),np.abs(1477-1482)])
dSoS_CA.append([np.abs(1470-1473),np.abs(1480-1486),np.abs(1500-1507),np.abs(1498-1508)])
dSoS_CA.append([np.abs(1484-1486),np.abs(1488-1492),np.abs(1462-1463),np.abs(1475-1476)])
dSoS_CA.append([np.abs(1484-1507),np.abs(1485-1496),np.abs(1481-1504),np.abs(1487-1513)])
dSoS_CA.append([np.abs(1471-1503),np.abs(1471-1493),np.abs(1476-1505),np.abs(1485-1511)])
dSoS_CA.append([np.abs(1464-1485),np.abs(1461-1482),np.abs(1466-1500),np.abs(1468-1505)])
dSoS_CA.append([np.abs(1483-1503),np.abs(1465-1487),np.abs(1475-1487),np.abs(1485-1502)])
dSoS_CA.append([np.abs(1514-1529),np.abs(1487-1504),np.abs(1549-1564),np.abs(1491-1524)])
dSoS_CA.append([np.abs(1502-1517),np.abs(1522-1530),np.abs(1510-1529),np.abs(1517-1536)])
dSoS_CA.append([np.abs(1494-1498),np.abs(1500-1515),np.abs(1498-1512),np.abs(1497-1513)])
dSoS_CA.append([np.abs(1463-1481),np.abs(1477-1498),np.abs(1473-1488),np.abs(1470-1491)])

np.save('/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/results/contrast_LBFGSdieter_CA.npy',dSoS_CA)
np.save('/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/results/contrast_LBFGSdieter_FA.npy',dSoS_FA)

#Contrast based on average SoS
dSoS_FA = []
dSoS_CA = []
dSoS_FA.append([np.abs(1496-1498),np.abs(1502-1504),np.abs(1502-1505),np.abs(1500-1512)])
dSoS_FA.append([np.abs(1485-1488),np.abs(1507-1509),np.abs(1505-1514),np.abs(1483-1489)])
dSoS_FA.append([np.abs(1480-1483),np.abs(1483-1491),np.abs(1485-1491),np.abs(1485-1491)])
dSoS_FA.append([np.abs(1518-1518),np.abs(1558-1563),np.abs(1534-1534),np.abs(1531-1535)])
dSoS_FA.append([np.abs(1512-1515),np.abs(1532-1538),np.abs(1532-1538),np.abs(1513-1516)])
dSoS_FA.append([np.abs(1519-1531),np.abs(1526-1529),np.abs(1517-1525),np.abs(1530-1531)])
dSoS_FA.append([np.abs(1534-1545),np.abs(1529-1533),np.abs(1514-1515),np.abs(1523-1531)])
dSoS_FA.append([np.abs(1539-1540),np.abs(1529-1534),np.abs(1515-1522),np.abs(1521-1526)])
dSoS_FA.append([np.abs(1520-1524),np.abs(1528-1535),np.abs(1523-1527),np.abs(1539-1543)])
dSoS_FA.append([np.abs(1537-1540),np.abs(1505-1506),np.abs(1533-1541),np.abs(1525-1529)])
dSoS_FA.append([np.abs(1508-1506),np.abs(1505-1509),np.abs(1531-1530),np.abs(1541-1548)])
dSoS_FA.append([np.abs(1546-1550),np.abs(1541-1546),np.abs(1543-1551),np.abs(1534-1543)])

dSoS_CA.append([np.abs(1470-1485),np.abs(1464-1486), np.abs(1466-1473),np.abs(1466-1474)])
dSoS_CA.append([np.abs(1476-1480),np.abs(1488-1490),np.abs(1491-1496),np.abs(1477-1479)])
dSoS_CA.append([np.abs(1470-1469),np.abs(1480-1480),np.abs(1500-1503),np.abs(1498-1504)])
dSoS_CA.append([np.abs(1484-1486),np.abs(1488-1491),np.abs(1462-1463),np.abs(1475-1476)])
dSoS_CA.append([np.abs(1484-1504),np.abs(1485-1495),np.abs(1481-1501),np.abs(1487-1510)])
dSoS_CA.append([np.abs(1471-1501),np.abs(1471-1489),np.abs(1476-1499),np.abs(1485-1509)])
dSoS_CA.append([np.abs(1464-1480),np.abs(1461-1479),np.abs(1466-1499),np.abs(1468-1505)])
dSoS_CA.append([np.abs(1483-1498),np.abs(1465-1484),np.abs(1475-1485),np.abs(1485-1497)])
dSoS_CA.append([np.abs(1514-1521),np.abs(1487-1502),np.abs(1549-1556),np.abs(1491-1520)])
dSoS_CA.append([np.abs(1502-1514),np.abs(1522-1524),np.abs(1510-1527),np.abs(1517-1529)])
dSoS_CA.append([np.abs(1494-1495),np.abs(1500-1506),np.abs(1498-1509),np.abs(1497-1502)])
dSoS_CA.append([np.abs(1463-1468),np.abs(1477-1493),np.abs(1473-1480),np.abs(1470-1492)])

np.save('/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/results/contrast_LBFGSdieter_CA_av.npy',dSoS_CA)
np.save('/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/results/contrast_LBFGSdieter_FA_av.npy',dSoS_FA)

#Contrast based on median SoS
dSoS_FA = []
dSoS_CA = []
dSoS_FA.append([np.abs(1496-1498),np.abs(1502-1503),np.abs(1502-1505),np.abs(1500-1513)])
dSoS_FA.append([np.abs(1485-1488),np.abs(1507-1509),np.abs(1505-1514),np.abs(1483-1489)])
dSoS_FA.append([np.abs(1480-1483),np.abs(1483-1492),np.abs(1485-1490),np.abs(1485-1488)])
dSoS_FA.append([np.abs(1518-1520),np.abs(1558-1562),np.abs(1534-1534),np.abs(1531-1535)])
dSoS_FA.append([np.abs(1512-1516),np.abs(1532-1538),np.abs(1532-1538),np.abs(1513-1516)])
dSoS_FA.append([np.abs(1519-1531),np.abs(1526-1528),np.abs(1517-1526),np.abs(1530-1532)])
dSoS_FA.append([np.abs(1534-1546),np.abs(1529-1532),np.abs(1514-1517),np.abs(1523-1532)])
dSoS_FA.append([np.abs(1539-1540),np.abs(1529-1534),np.abs(1515-1523),np.abs(1521-1527)])
dSoS_FA.append([np.abs(1520-1525),np.abs(1528-1535),np.abs(1523-1527),np.abs(1539-1544)])
dSoS_FA.append([np.abs(1537-1540),np.abs(1505-1506),np.abs(1533-1541),np.abs(1525-1529)])
dSoS_FA.append([np.abs(1508-1506),np.abs(1505-1509),np.abs(1531-1529),np.abs(1541-1547)])
dSoS_FA.append([np.abs(1546-1550),np.abs(1541-1545),np.abs(1543-1550),np.abs(1534-1543)])

dSoS_CA.append([np.abs(1470-1485),np.abs(1464-1486), np.abs(1466-1472),np.abs(1466-1474)])
dSoS_CA.append([np.abs(1476-1479),np.abs(1488-1489),np.abs(1491-1495),np.abs(1477-1479)])
dSoS_CA.append([np.abs(1470-1468),np.abs(1480-1479),np.abs(1500-1502),np.abs(1498-1505)])
dSoS_CA.append([np.abs(1484-1486),np.abs(1488-1491),np.abs(1462-1462),np.abs(1475-1475)])
dSoS_CA.append([np.abs(1484-1504),np.abs(1485-1496),np.abs(1481-1501),np.abs(1487-1512)])
dSoS_CA.append([np.abs(1471-1501),np.abs(1471-1490),np.abs(1476-1502),np.abs(1485-1509)])
dSoS_CA.append([np.abs(1464-1483),np.abs(1461-1480),np.abs(1466-1500),np.abs(1468-1505)])
dSoS_CA.append([np.abs(1483-1499),np.abs(1465-1484),np.abs(1475-1486),np.abs(1485-1497)])
dSoS_CA.append([np.abs(1514-1521),np.abs(1487-1503),np.abs(1549-1555),np.abs(1491-1523)])
dSoS_CA.append([np.abs(1502-1515),np.abs(1522-1524),np.abs(1510-1526),np.abs(1517-1526)])
dSoS_CA.append([np.abs(1494-1494),np.abs(1500-1508),np.abs(1498-1509),np.abs(1497-1499)])
dSoS_CA.append([np.abs(1463-1468),np.abs(1477-1492),np.abs(1473-1480),np.abs(1470-1491)])

np.save('/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/results/contrast_LBFGSdieter_CA_med.npy',dSoS_CA)
np.save('/scratch_net/biwidl307/sonia/USImageReconstruction-Melanie/results/contrast_LBFGSdieter_FA_med.npy',dSoS_FA)