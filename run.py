#!/usr/bin/env python
# -*- coding: utf-8 -*-

from eon_simulador import Simulador
import simpy
from random import *
from config import *
import numpy as np

def Calculainterval(sample):
	# calcula média e interval de confiança de uma amostra (t de Student) 95%. 
	# calculates mean and one-sample confidence interval (Student's t) 95%.
	media = np.mean(sample)
	desvio = np.std(sample, ddof=1)
	interval = (desvio/len(sample))*1.833
	return [media,interval]

def main(args):
	topology = TOPOLOGY
	file1  = open('out/'+topology+'/Block'+'.dat', 'w')
	file2  = open('out/'+topology+'/Block_10'+'.dat', 'w')
	file3  = open('out/'+topology+'/Block_20'+'.dat', 'w')
	file4  = open('out/'+topology+'/Block_40'+'.dat', 'w')
	file5  = open('out/'+topology+'/Block_80'+'.dat', 'w')
	file6  = open('out/'+topology+'/Block_160'+'.dat', 'w')
	file7  = open('out/'+topology+'/Block_200'+'.dat', 'w')
	file8  = open('out/'+topology+'/Block_400'+'.dat', 'w')
	file9  = open('out/'+topology+'/Block_classe1'+'.dat', 'w')
	file10  = open('out/'+topology+'/Block_classe2'+'.dat', 'w')
	file11  = open('out/'+topology+'/Block_classe3'+'.dat', 'w')
	file12  = open('out/'+topology+'/Block_banda'+'.dat', 'w')

	for e in range(ERLANG_MIN, ERLANG_MAX+1, ERLANG_INC):
		Block = []
		Block_10 = []
		Block_20 = []
		Block_40 = []
		Block_80 = []
		Block_160 = []
		Block_200 = []
		Block_400 = []
		Block_classe1 = []
		Block_classe2 = []
		Block_classe3 = []
		Block_banda = []

		for rep in range(10):
			rate = e / HOLDING_TIME
			seed(RANDOM_SEED[rep])
			env = simpy.Environment()
			simulador = Simulador(env)
			env.process(simulador.Run(rate))
			env.run()
			print("Erlang", e, "Simulation...", rep)
			print("blocked", simulador.NumReqBlocked, "de", NUM_OF_REQUESTS)
			Block.append(simulador.NumReqBlocked / float(NUM_OF_REQUESTS))
			Block_10.append(simulador.NumReqBlocked_10/float(simulador.NumReq_10))
			Block_20.append(simulador.NumReqBlocked_20/float(simulador.NumReq_20))
			Block_40.append(simulador.NumReqBlocked_40/float(simulador.NumReq_40))
			Block_80.append(simulador.NumReqBlocked_80/float(simulador.NumReq_80))
			Block_160.append(simulador.NumReqBlocked_160/float(simulador.NumReq_160))
			Block_200.append(simulador.NumReqBlocked_200/float(simulador.NumReq_200))
			Block_400.append(simulador.NumReqBlocked_400/float(simulador.NumReq_400))
			Block_classe1.append(simulador.NumReqBlocked_classe1/float(simulador.NumReq_classe1))
			Block_classe2.append(simulador.NumReqBlocked_classe2/float(simulador.NumReq_classe2))
			Block_classe3.append(simulador.NumReqBlocked_classe3/float(simulador.NumReq_classe3))
			#DB_requested
			BD_solicitada = ((simulador.NumReq_10)*10+(simulador.NumReq_20)*20+(simulador.NumReq_40)*40+(simulador.NumReq_80)*80+(simulador.NumReq_160)*160+(simulador.NumReq_200)*200+(simulador.NumReq_400)*400)
			BD_bloqueada = ((simulador.NumReqBlocked_10)*10+(simulador.NumReqBlocked_20)*20+(simulador.NumReqBlocked_40)*40+(simulador.NumReqBlocked_80)*80+(simulador.NumReqBlocked_160)*160+(simulador.NumReqBlocked_200)*200+(simulador.NumReqBlocked_400)*400)
			Block_banda.append(BD_bloqueada/float(BD_solicitada))

		interval = Calculainterval(Block)
		interval_10 = Calculainterval(Block_10)
		interval_20 = Calculainterval(Block_20)
		interval_40 = Calculainterval(Block_40)
		interval_80 = Calculainterval(Block_80)
		interval_160 = Calculainterval(Block_160)
		interval_200 = Calculainterval(Block_200)
		interval_400 = Calculainterval(Block_400)
		interval_classe1 = Calculainterval(Block_classe1)
		interval_classe2 = Calculainterval(Block_classe2)
		interval_classe3 = Calculainterval(Block_classe3)
		interval_Block_banda = Calculainterval(Block_banda)

		file1.write(str(e))
		file1.write("\t")
		file1.write(str(interval[0]))
		file1.write("\t")
		file1.write(str(interval[0]-interval[1]))
		file1.write("\t")
		file1.write(str(interval[0]+interval[1]))
		file1.write("\n")

		file2.write(str(e))
		file2.write("\t")
		file2.write(str(interval_10[0]))
		file2.write("\t")
		file2.write(str(interval_10[0]-interval_10[1]))
		file2.write("\t")
		file2.write(str(interval_10[0]+interval_10[1]))
		file2.write("\n")

		file3.write(str(e))
		file3.write("\t")
		file3.write(str(interval_20[0]))
		file3.write("\t")
		file3.write(str(interval_20[0]-interval_20[1]))
		file3.write("\t")
		file3.write(str(interval_20[0]+interval_20[1]))
		file3.write("\n")

		file4.write(str(e))
		file4.write("\t")
		file4.write(str(interval_40[0]))
		file4.write("\t")
		file4.write(str(interval_40[0]-interval_40[1]))
		file4.write("\t")
		file4.write(str(interval_40[0]+interval_40[1]))
		file4.write("\n")

		file5.write(str(e))
		file5.write("\t")
		file5.write(str(interval_80[0]))
		file5.write("\t")
		file5.write(str(interval_80[0]-interval_80[1]))
		file5.write("\t")
		file5.write(str(interval_80[0]+interval_80[1]))
		file5.write("\n")

		file6.write(str(e))
		file6.write("\t")
		file6.write(str(interval_160[0]))
		file6.write("\t")
		file6.write(str(interval_160[0]-interval_160[1]))
		file6.write("\t")
		file6.write(str(interval_160[0]+interval_160[1]))
		file6.write("\n")

		file7.write(str(e))
		file7.write("\t")
		file7.write(str(interval_200[0]))
		file7.write("\t")
		file7.write(str(interval_200[0]-interval_200[1]))
		file7.write("\t")
		file7.write(str(interval_200[0]+interval_200[1]))
		file7.write("\n")

		file8.write(str(e))
		file8.write("\t")
		file8.write(str(interval_400[0]))
		file8.write("\t")
		file8.write(str(interval_400[0]-interval_400[1]))
		file8.write("\t")
		file8.write(str(interval_400[0]+interval_400[1]))
		file8.write("\n")

		file9.write(str(e))
		file9.write("\t")
		file9.write(str(interval_classe1[0]))
		file9.write("\t")
		file9.write(str(interval_classe1[0]-interval_classe1[1]))
		file9.write("\t")
		file9.write(str(interval_classe1[0]+interval_classe1[1]))
		file9.write("\n")

		file10.write(str(e))
		file10.write("\t")
		file10.write(str(interval_classe2[0]))
		file10.write("\t")
		file10.write(str(interval_classe2[0]-interval_classe2[1]))
		file10.write("\t")
		file10.write(str(interval_classe2[0]+interval_classe2[1]))
		file10.write("\n")

		file11.write(str(e))
		file11.write("\t")
		file11.write(str(interval_classe3[0]))
		file11.write("\t")
		file11.write(str(interval_classe3[0]-interval_classe3[1]))
		file11.write("\t")
		file11.write(str(interval_classe3[0]+interval_classe3[1]))
		file11.write("\n")

		file12.write(str(e))
		file12.write("\t")
		file12.write(str(interval_Block_banda[0]))
		file12.write("\t")
		file12.write(str(interval_Block_banda[0]-interval_Block_banda[1]))
		file12.write("\t")
		file12.write(str(interval_Block_banda[0]+interval_Block_banda[1]))
		file12.write("\n")

	file1.close()
	file2.close()
	file3.close()
	file4.close()
	file5.close()
	file6.close()
	file7.close()
	file8.close()
	file10.close()
	file11.close()
	file12.close()

	return 0

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
