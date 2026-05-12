# GSolve - gravity processing software.
# Copyright (c) 2026 Earth Sciences New Zealand.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# SPDX-License-Identifier: GPLv3

import pathlib

import pandas as pd
import pandas.testing as pdt
import pytest

from gsolve.core.utils import to_naive_utc_datetime
from gsolve.scintrex import CG6Data

cg6_data = """\
/		CG-6 Survey
/		Survey Name:	GSOLVE
/		Instrument Serial Number:	000000025010680
/		Created:	2025-02-18 20:28:54
/
/		CG-6 Calibration
/		Operator:	CM
/		Gcal1 [mGal]:	7850.406000
/		Goff [ADU]:	-8388608.000000
/		Gref [mGal]:	0.0000
/		X Scale [arc-sec/ADU]:	0.045186
/		Y Scale [arc-sec/ADU]:	0.045305
/		X Offset [ADU]:	-22307.000000
/		Y Offset [ADU]:	20920.000000
/		Temperature Coefficient [mGal/mK]:	-0.121000
/		Temperature Scale [mK/ADU]:	-0.000111
/		Drift Rate [mGal/day]:	0.095080
/		Drift Zero Time:	2025-01-24 18:00:00
/		Firmware Version:	CG6_2_20240409
/
/Station	Date	Time	CorrGrav	Line	StdDev	StdErr	RawGrav	X	Y	SensorTemp	TideCorr	TiltCorr	TempCorr	DriftCorr	MeasurDur	InstrHeight	LatUser	LonUser	ElevUser	LatGPS	LonGPS	ElevGPS	Corrections[drift-temp-na-tide-tilt]
GNS_LAB_WAIRAKEI	2025-02-18	20:28:54	3973.6465	1	0.0553	0.0071	3975.8542	-4.3	2.8	1.4888	-0.0014	0.0003	0.1802	-2.3868	60	0.000	-38.632431	176.093964	388.90	-38.632420	176.094025	386.6	11011
GNS_LAB_WAIRAKEI	2025-02-18	20:29:54	3973.6469	1	0.0599	0.0077	3975.8552	-4.6	2.6	1.4876	-0.0017	0.0003	0.1800	-2.3869	60	0.000	-38.632431	176.093964	388.90	-38.632401	176.094040	386.6	11011
GNS_LAB_WAIRAKEI	2025-02-18	20:30:54	3973.6470	1	0.0551	0.0071	3975.8557	-4.4	2.4	1.4861	-0.0020	0.0003	0.1799	-2.3869	60	0.000	-38.632431	176.093964	388.90	-38.632401	176.094025	386.6	11011
GNS_LAB_WAIRAKEI	2025-02-18	20:31:54	3973.6458	1	0.0512	0.0066	3975.8547	-4.9	2.4	1.4864	-0.0022	0.0004	0.1799	-2.3870	60	0.000	-38.632431	176.093964	388.90	-38.632393	176.094025	386.9	11011
GNS_LAB_WAIRAKEI	2025-02-18	20:32:54	3973.6482	1	0.0558	0.0072	3975.8573	-5.3	2.3	1.4869	-0.0025	0.0004	0.1800	-2.3871	60	0.000	-38.632431	176.093964	388.90	-38.632401	176.094025	387.2	11011
GNS_LAB_WAIRAKEI	2025-02-18	20:33:54	3973.6474	1	0.0504	0.0065	3975.8571	-5.0	2.3	1.4863	-0.0027	0.0004	0.1797	-2.3871	60	0.000	-38.632431	176.093964	388.90	-38.632431	176.094025	387.3	11011
GNS_LAB_WAIRAKEI	2025-02-18	20:34:54	3973.6472	1	0.0483	0.0062	3975.8572	-5.3	2.0	1.4855	-0.0030	0.0004	0.1798	-2.3872	60	0.000	-38.632431	176.093964	388.90	-38.632458	176.093964	387.0	11011
GNS_LAB_WAIRAKEI	2025-02-18	20:35:54	3973.6461	1	0.0443	0.0057	3975.8565	-5.3	2.2	1.4855	-0.0032	0.0004	0.1796	-2.3873	60	0.000	-38.632431	176.093964	388.90	-38.632465	176.093948	387.3	11011
GNS_LAB_WAIRAKEI	2025-02-18	20:36:54	3973.6468	1	0.0520	0.0067	3975.8574	-5.2	1.7	1.4861	-0.0035	0.0004	0.1798	-2.3873	60	0.000	-38.632431	176.093964	388.90	-38.632477	176.093964	387.4	11011
GNS_LAB_WAIRAKEI	2025-02-18	20:37:54	3973.6465	1	0.0517	0.0067	3975.8575	-5.2	1.6	1.4860	-0.0037	0.0003	0.1797	-2.3874	60	0.000	-38.632431	176.093964	388.90	-38.632477	176.093964	388.3	11011
GNS_LAB_WAIRAKEI	2025-02-18	20:38:54	3973.6466	1	0.0482	0.0062	3975.8579	-5.2	1.5	1.4857	-0.0039	0.0004	0.1798	-2.3875	60	0.000	-38.632431	176.093964	388.90	-38.632439	176.094025	388.1	11011
GNS_LAB_WAIRAKEI	2025-02-18	20:39:54	3973.6476	1	0.0527	0.0068	3975.8591	-5.3	1.4	1.4860	-0.0042	0.0004	0.1799	-2.3875	60	0.000	-38.632431	176.093964	388.90	-38.632458	176.093964	389.6	11011
GNS_LAB_WAIRAKEI	2025-02-18	20:40:54	3973.6460	1	0.0539	0.0070	3975.8580	-5.2	1.5	1.4850	-0.0044	0.0003	0.1797	-2.3876	60	0.000	-38.632431	176.093964	388.90	-38.632477	176.093918	390.0	11011
GNS_LAB_WAIRAKEI	2025-02-18	20:41:54	3973.6482	1	0.0563	0.0073	3975.8603	-5.1	1.5	1.4854	-0.0047	0.0004	0.1798	-2.3877	60	0.000	-38.632431	176.093964	388.90	-38.632420	176.093964	390.6	11011
GNS_LAB_WAIRAKEI	2025-02-18	20:42:54	3973.6480	1	0.0484	0.0063	3975.8606	-5.5	1.3	1.4851	-0.0049	0.0004	0.1797	-2.3877	60	0.000	-38.632431	176.093964	388.90	-38.632439	176.093948	392.1	11011
WAIRAKEI_ABS	2025-02-18	20:57:50	3973.8692	1	0.0691	0.0089	3976.0845	-2.2	1.7	1.5005	-0.0082	0.0001	0.1815	-2.3887	60	0.000	-38.632572	176.093658	394.70	-38.632584	176.093613	393.8	11011
WAIRAKEI_ABS	2025-02-18	20:58:50	3973.8704	1	0.0565	0.0073	3976.0858	-1.6	2.0	1.5012	-0.0084	0.0001	0.1816	-2.3888	60	0.000	-38.632572	176.093658	394.70	-38.632591	176.093613	394.0	11011
WAIRAKEI_ABS	2025-02-18	20:59:50	3973.8713	1	0.0558	0.0072	3976.0868	-1.1	2.3	1.5027	-0.0086	0.0001	0.1819	-2.3888	60	0.000	-38.632572	176.093658	394.70	-38.632580	176.093658	394.6	11011
WAIRAKEI_ABS	2025-02-18	21:00:50	3973.8704	1	0.0615	0.0079	3976.0863	-1.0	2.1	1.5016	-0.0088	0.0001	0.1817	-2.3889	60	0.000	-38.632572	176.093658	394.70	-38.632591	176.093689	395.3	11011
WAIRAKEI_ABS	2025-02-18	21:01:50	3973.8706	1	0.0702	0.0091	3976.0867	-0.7	2.4	1.5027	-0.0090	0.0001	0.1818	-2.3890	60	0.000	-38.632572	176.093658	394.70	-38.632545	176.093689	396.8	11011
WAIRAKEI_ABS	2025-02-18	21:02:50	3973.8695	1	0.0628	0.0081	3976.0857	-0.3	2.9	1.5032	-0.0092	0.0001	0.1819	-2.3890	60	0.000	-38.632572	176.093658	394.70	-38.632553	176.093689	396.8	11011
WAIRAKEI_ABS	2025-02-18	21:03:50	3973.8715	1	0.1121	0.0145	3976.0880	-1.4	2.3	1.5034	-0.0094	0.0001	0.1819	-2.3891	60	0.000	-38.632572	176.093658	394.70	-38.632591	176.093613	397.0	11011
WAIRAKEI_ABS	2025-02-18	21:04:50	3973.8710	1	0.0719	0.0093	3976.0876	-1.6	2.5	1.5037	-0.0096	0.0001	0.1821	-2.3892	60	0.000	-38.632572	176.093658	394.70	-38.632603	176.093658	397.7	11011
WAIRAKEI_ABS	2025-02-18	21:05:50	3973.8716	1	0.0614	0.0079	3976.0885	-2.2	1.7	1.5045	-0.0098	0.0001	0.1821	-2.3892	60	0.000	-38.632572	176.093658	394.70	-38.632603	176.093582	398.5	11011
WAIRAKEI_ABS	2025-02-18	21:06:50	3973.8684	1	0.0696	0.0090	3976.0853	-5.3	1.0	1.5042	-0.0100	0.0003	0.1820	-2.3893	60	0.000	-38.632572	176.093658	394.70	-38.632526	176.093613	399.9	11011
WAIRAKEI_ABS	2025-02-18	21:07:50	3973.8681	1	0.0699	0.0090	3976.0853	-4.9	0.9	1.5036	-0.0102	0.0003	0.1820	-2.3894	60	0.000	-38.632572	176.093658	394.70	-38.632515	176.093689	400.8	11011
WAIRAKEI_ABS	2025-02-18	21:08:50	3973.8696	1	0.0596	0.0077	3976.0872	-3.7	1.3	1.5040	-0.0103	0.0002	0.1820	-2.3894	60	0.000	-38.632572	176.093658	394.70	-38.632469	176.093613	403.0	11011
WAIRAKEI_ABS	2025-02-18	21:09:50	3973.8694	1	0.0731	0.0094	3976.0871	-2.7	2.0	1.5057	-0.0105	0.0001	0.1822	-2.3895	60	0.000	-38.632572	176.093658	394.70	-38.632507	176.093613	402.9	11011
WAIRAKEI_ABS	2025-02-18	21:10:50	3973.8711	1	0.0461	0.0059	3976.0890	-2.6	2.1	1.5070	-0.0107	0.0001	0.1823	-2.3896	60	0.000	-38.632572	176.093658	394.70	-38.632584	176.093613	403.4	11011
WAIRAKEI_ABS	2025-02-18	21:11:50	3973.8687	1	0.1579	0.0204	3976.0868	-2.2	1.9	1.5071	-0.0109	0.0001	0.1823	-2.3896	60	0.000	-38.632572	176.093658	394.70	-38.632561	176.093582	403.7	11011
GNS_LAB_WAIRAKEI	2025-02-18	21:18:23	3973.6477	1	0.0516	0.0067	3975.8669	-3.9	1.8	1.5102	-0.0120	0.0002	0.1827	-2.3901	60	0.000	-38.632507	176.093796	399.50	-38.632389	176.093796	400.6	11011
GNS_LAB_WAIRAKEI	2025-02-18	21:19:23	3973.6477	1	0.0620	0.0080	3975.8670	-3.7	1.8	1.5102	-0.0122	0.0002	0.1828	-2.3901	60	0.000	-38.632507	176.093796	399.50	-38.632420	176.093842	395.3	11011
GNS_LAB_WAIRAKEI	2025-02-18	21:20:23	3973.6474	1	0.0463	0.0060	3975.8671	-3.6	1.7	1.5093	-0.0123	0.0002	0.1827	-2.3902	60	0.000	-38.632507	176.093796	399.50	-38.632374	176.094025	393.5	11011
GNS_LAB_WAIRAKEI	2025-02-18	21:21:23	3973.6484	1	0.0494	0.0064	3975.8682	-3.6	1.7	1.5095	-0.0125	0.0002	0.1828	-2.3903	60	0.000	-38.632507	176.093796	399.50	-38.632401	176.093887	392.7	11011
GNS_LAB_WAIRAKEI	2025-02-18	21:22:23	3973.6483	1	0.0475	0.0061	3975.8686	-3.5	1.4	1.5083	-0.0127	0.0002	0.1825	-2.3903	60	0.000	-38.632507	176.093796	399.50	-38.632374	176.093948	393.2	11011
GNS_LAB_WAIRAKEI	2025-02-18	21:23:23	3973.6510	1	0.0416	0.0054	3975.8717	-3.3	1.0	1.5074	-0.0128	0.0001	0.1824	-2.3904	60	0.000	-38.632507	176.093796	399.50	-38.632401	176.093918	391.1	11011
GNS_LAB_WAIRAKEI	2025-02-18	21:24:23	3973.6506	1	0.0476	0.0061	3975.8716	-3.0	0.9	1.5067	-0.0130	0.0001	0.1823	-2.3905	60	0.000	-38.632507	176.093796	399.50	-38.632401	176.093918	390.1	11011
GNS_LAB_WAIRAKEI	2025-02-18	21:25:23	3973.6511	1	0.0554	0.0072	3975.8725	-3.0	0.9	1.5062	-0.0131	0.0001	0.1821	-2.3905	60	0.000	-38.632507	176.093796	399.50	-38.632477	176.093964	386.5	11011
GNS_LAB_WAIRAKEI	2025-02-18	21:26:23	3973.6504	1	0.0551	0.0071	3975.8721	-3.0	1.0	1.5051	-0.0133	0.0001	0.1821	-2.3906	60	0.000	-38.632507	176.093796	399.50	-38.632355	176.093964	388.9	11011
GNS_LAB_WAIRAKEI	2025-02-18	21:27:23	3973.6507	1	0.0455	0.0059	3975.8724	-2.5	0.7	1.5052	-0.0134	0.0001	0.1822	-2.3907	60	0.000	-38.632507	176.093796	399.50	-38.632420	176.093964	387.9	11011
GNS_LAB_WAIRAKEI	2025-02-18	21:28:23	3973.6498	1	0.0506	0.0065	3975.8722	-2.0	0.8	1.5037	-0.0136	0.0001	0.1818	-2.3907	60	0.000	-38.632507	176.093796	399.50	-38.632431	176.093948	388.2	11011
GNS_LAB_WAIRAKEI	2025-02-18	21:29:23	3973.6501	1	0.0548	0.0071	3975.8726	-2.2	0.5	1.5029	-0.0137	0.0001	0.1819	-2.3908	60	0.000	-38.632507	176.093796	399.50	-38.632523	176.093842	388.7	11011
GNS_LAB_WAIRAKEI	2025-02-18	21:30:23	3973.6509	1	0.0489	0.0063	3975.8737	-2.2	0.3	1.5029	-0.0138	0.0001	0.1818	-2.3909	60	0.000	-38.632507	176.093796	399.50	-38.632504	176.093719	387.2	11011
GNS_LAB_WAIRAKEI	2025-02-18	21:31:23	3973.6492	1	0.0516	0.0067	3975.8723	-2.1	0.2	1.5019	-0.0140	0.0001	0.1817	-2.3909	60	0.000	-38.632507	176.093796	399.50	-38.632507	176.093887	389.0	11011
GNS_LAB_WAIRAKEI	2025-02-18	21:32:23	3973.6500	1	0.0630	0.0081	3975.8736	-1.9	0.3	1.5003	-0.0141	0.0001	0.1815	-2.3910	60	0.000	-38.632507	176.093796	399.50	-38.632381	176.093964	391.5	11011
GNS_LAB_WAIRAKEI	2025-02-19	20:02:14	3973.5619	2	0.0352	0.0045	3975.8238	-0.3	-2.7	1.4913	0.0376	0.0001	0.1805	-2.4801	60	0.000	-38.632591	176.094025	397.60	-38.632439	176.094025	389.0	11011
GNS_LAB_WAIRAKEI	2025-02-19	20:03:14	3973.5616	2	0.0363	0.0047	3975.8239	-0.3	-3.1	1.4917	0.0373	0.0001	0.1804	-2.4802	60	0.000	-38.632591	176.094025	397.60	-38.632450	176.094101	388.6	11011
GNS_LAB_WAIRAKEI	2025-02-19	20:04:14	3973.5624	2	0.0343	0.0044	3975.8252	-0.5	-3.1	1.4910	0.0370	0.0001	0.1803	-2.4802	60	0.000	-38.632591	176.094025	397.60	-38.632439	176.094025	386.8	11011
GNS_LAB_WAIRAKEI	2025-02-19	20:05:14	3973.5633	2	0.0396	0.0051	3975.8263	-0.5	-3.1	1.4909	0.0368	0.0001	0.1804	-2.4803	60	0.000	-38.632591	176.094025	397.60	-38.632507	176.093918	387.0	11011
GNS_LAB_WAIRAKEI	2025-02-19	20:06:14	3973.5643	2	0.1068	0.0138	3975.8280	-0.6	-2.4	1.4899	0.0365	0.0001	0.1802	-2.4804	60	0.000	-38.632591	176.094025	397.60	-38.632450	176.093964	387.8	11011
GNS_LAB_WAIRAKEI	2025-02-19	20:07:14	3973.5632	2	0.0770	0.0099	3975.8269	-0.3	-3.0	1.4900	0.0362	0.0001	0.1803	-2.4804	60	0.000	-38.632591	176.094025	397.60	-38.632484	176.093918	388.1	11011
GNS_LAB_WAIRAKEI	2025-02-19	20:08:14	3973.5621	2	0.0567	0.0073	3975.8264	-1.0	-2.2	1.4897	0.0359	0.0001	0.1802	-2.4805	60	0.000	-38.632591	176.094025	397.60	-38.632458	176.094025	389.5	11011
WAIRAKEI_ABS	2025-02-19	20:14:06	3973.7779	2	0.0488	0.0063	3976.0440	-6.9	0.8	1.4876	0.0343	0.0006	0.1800	-2.4809	60	0.000	-38.632565	176.093582	397.20	-38.632565	176.093582	397.8	11011
WAIRAKEI_ABS	2025-02-19	20:15:06	3973.7803	2	0.0393	0.0051	3976.0466	-7.0	0.6	1.4883	0.0340	0.0005	0.1801	-2.4810	60	0.000	-38.632565	176.093582	397.20	-38.632561	176.093582	398.0	11011
WAIRAKEI_ABS	2025-02-19	20:16:06	3973.7780	2	0.0429	0.0055	3976.0448	-7.6	0.8	1.4875	0.0337	0.0007	0.1799	-2.4810	60	0.000	-38.632565	176.093582	397.20	-38.632565	176.093582	398.1	11011
WAIRAKEI_ABS	2025-02-19	20:17:06	3973.7776	2	0.0493	0.0064	3976.0447	-7.8	0.5	1.4873	0.0334	0.0007	0.1799	-2.4811	60	0.000	-38.632565	176.093582	397.20	-38.632572	176.093582	398.2	11011
WAIRAKEI_ABS	2025-02-19	20:18:06	3973.7757	2	0.0351	0.0045	3976.0433	-7.5	0.1	1.4865	0.0331	0.0007	0.1798	-2.4812	60	0.000	-38.632565	176.093582	397.20	-38.632572	176.093658	398.0	11011
WAIRAKEI_ABS	2025-02-19	20:19:06	3973.7781	2	0.0487	0.0063	3976.0459	-7.6	0.3	1.4865	0.0328	0.0007	0.1799	-2.4812	60	0.000	-38.632565	176.093582	397.20	-38.632565	176.093613	398.4	11011
GNS_LAB_WAIRAKEI	2025-02-19	20:22:44	3973.5653	2	0.0531	0.0069	3975.8347	5.7	-0.8	1.4858	0.0318	0.0004	0.1798	-2.4815	60	0.000	-38.632420	176.094025	393.50	-38.632431	176.094025	393.6	11011
GNS_LAB_WAIRAKEI	2025-02-19	20:23:44	3973.5672	2	0.0355	0.0046	3975.8369	6.0	-1.7	1.4857	0.0315	0.0005	0.1798	-2.4815	60	0.000	-38.632420	176.094025	393.50	-38.632431	176.094025	393.5	11011
GNS_LAB_WAIRAKEI	2025-02-19	20:24:44	3973.5684	2	0.0346	0.0045	3975.8387	5.5	-1.8	1.4852	0.0312	0.0004	0.1797	-2.4816	60	0.000	-38.632420	176.094025	393.50	-38.632450	176.093964	393.6	11011
GNS_LAB_WAIRAKEI	2025-02-19	20:25:44	3973.5667	2	0.0640	0.0083	3975.8374	5.0	-1.7	1.4855	0.0310	0.0003	0.1796	-2.4817	60	0.000	-38.632420	176.094025	393.50	-38.632439	176.093964	394.0	11011
GNS_LAB_WAIRAKEI	2025-02-19	20:26:44	3973.5684	2	0.0633	0.0082	3975.8395	4.2	-1.0	1.4854	0.0307	0.0002	0.1797	-2.4817	60	0.000	-38.632420	176.094025	393.50	-38.632393	176.093964	394.6	11011
"""


@pytest.fixture
def cg6_file(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    file = tmp_path_factory.mktemp("data") / "test.cg6"
    with open(file, "w") as f:
        f.write(cg6_data)
    return file


@pytest.fixture
def sample_data(
    cg6_file: pytest.TempPathFactory, loop_from_line: bool = True
) -> CG6Data:
    return CG6Data.from_file(cg6_file, loop_from_line=loop_from_line)


def test_cg6data_initialization(sample_data: CG6Data) -> None:
    cg6copy = sample_data.copy()
    cg6init = CG6Data(
        sample_data.data, sample_data.metadata, sample_data.metadata_units
    )
    for cg6 in [cg6copy, cg6init]:
        pdt.assert_frame_equal(cg6.data, sample_data.data)
        assert all(
            [cg6.metadata[k] == sample_data.metadata[k] for k in cg6.metadata.keys()]
        )
        assert all(
            [
                cg6.metadata_units[k] == sample_data.metadata_units[k]
                for k in cg6.metadata_units.keys()
            ]
        )


def test_cg6data_set_loop(cg6_file: pathlib.Path) -> None:
    # check that loop is read from field
    cg6 = CG6Data.from_file(cg6_file, loop_from_line=False)
    assert "loop" not in cg6.data.columns

    cg6_with_loop = CG6Data.from_file(cg6_file, loop_from_line=True)
    assert "loop" in cg6_with_loop.data.columns
    pdt.assert_series_equal(
        cg6_with_loop.data["line"],
        cg6_with_loop.data["loop"],
        check_index=False,
        check_names=False,
    )

    # set from field
    cg6.set_loop(field="line")
    assert "loop" in cg6.data.columns
    with pytest.raises(KeyError):
        cg6.set_loop(field="xxxx")
    with pytest.raises(ValueError):
        cg6.set_loop(field="line", time_gap="1h")
        cg6.set_loop()

    # Case 1: set from array
    cg6.set_loop(array=["xxx"] * cg6.data.shape[0])
    assert cg6.data["loop"].eq("xxx").all()
    with pytest.raises(ValueError):
        cg6.set_loop(array=[1, 2, 3])

    # Case 2: set from datetimes

    j_dt = cg6.data.columns.get_loc("datetime")
    i_mid = cg6.data.shape[0] // 2
    times = [cg6.data.iat[0, j_dt], cg6.data.iat[i_mid, j_dt]]

    # 2.1: list of datetimes

    cg6.set_loop(datetimes=times, loop_start=100, loop_step=2)
    assert cg6.data["loop"].iloc[:i_mid].eq("100").all()
    assert cg6.data["loop"].iloc[i_mid:].eq("102").all()

    # 2.2 from a dict of datetimes
    cg6.set_loop(
        datetimes=dict(zip(times, ["a", "b"])), loop_start=200, output_column="xloop"
    )
    assert cg6.data["xloop"].iat[0] == "a"
    assert cg6.data["xloop"].iat[i_mid] == "b"

    # 2.3 from a series
    cg6.set_loop(
        datetimes=pd.Series(index=to_naive_utc_datetime(times), data=["y", "z"])
    )
    assert cg6.data["loop"].iat[0] == "y"
    assert cg6.data["loop"].iat[i_mid] == "z"

    # Case 3: set from time gap
    fstr = "a_{LOOP}"
    cg6.set_loop(
        time_gap="12h", loop_start=300, loop_format=fstr, output_column="zloop"
    )
    assert cg6.data["zloop"].iat[0] == fstr.format(LOOP=300)
    assert cg6.data.loc[cg6.data["line"].eq(2), "zloop"].eq(fstr.format(LOOP=301)).all()

    # Case X: bad args
    with pytest.raises(TypeError):
        # not an array like
        cg6.set_loop(datetimes="2025-02-18 20:36:54")
    with pytest.raises(ValueError):
        # not sorted increasing
        cg6.set_loop(datetimes=list(reversed(times)), loop_start=100)
    with pytest.raises(ValueError):
        # not sorted increasing
        times[0] = cg6.data.iat[1, j_dt] + pd.Timedelta("1h")
        cg6.set_loop(datetimes=times)
    with pytest.raises(ValueError, match="LOOP"):
        # bad format string
        cg6.set_loop(datetimes=times, loop_format="a_{}", loop_start=100)


def test_set_drift_correction(sample_data: CG6Data) -> None:
    "drift_rate"
    "drift_zero_time"
    drate = 0.1
    dzero = sample_data.data["datetime"].min()
    assert sample_data.metadata["drift_rate"] != drate
    data2 = sample_data.copy()
    data2.set_drift_correction(drift_rate=drate, drift_zero_time=dzero)

    assert data2.metadata["drift_rate"] == drate
    assert data2.metadata["drift_zero_time"] == dzero
    # pdt.assert_series_equal(
    #     updated.data["driftcorr"],
    #     updated.data["datetime"]
    #     .sub(dzero)
    #     .dt.total_seconds()
    #     .astype(float)
    #     .div(86400)
    #     .mul(-1 * drate),
    #     check_names=False,
    # )


def test_cg6data_meter_id(sample_data: CG6Data) -> None:
    assert sample_data.meter_id == "0680"
    sample_data.metadata["instrument_serial_number"] = "1234x"
    assert sample_data.meter_id == "234x"
    sample_data.metadata["instrument_serial_number"] = "1"
    assert sample_data.meter_id == "1"


def test_cg6data_stations(sample_data: CG6Data) -> None:
    assert sorted(sample_data.stations) == sorted(["GNS_LAB_WAIRAKEI", "WAIRAKEI_ABS"])


def test_cg6data_to_gsolve_observations(sample_data: CG6Data) -> None:
    observations = sample_data.to_gsolve_observations()

    assert observations is not None
    assert "meter_reading" in observations.data.columns
    assert "meter_reading_mgal" in observations.data.columns
    sample_data.data.drop(columns="loop", inplace=True)
    with pytest.raises(ValueError):
        sample_data.to_gsolve_observations()


def test_cg6data_to_gsolve_sites(sample_data: CG6Data) -> None:
    sites_user = sample_data.to_gsolve_sites(coords_source="user")
    sites_gps = sample_data.to_gsolve_sites(coords_source="gps")

    assert sites_user is not None
    assert sorted(sites_user.data.index.to_list()) == [
        "GNS_LAB_WAIRAKEI",
        "WAIRAKEI_ABS",
    ]
