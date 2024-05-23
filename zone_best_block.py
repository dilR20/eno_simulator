import copy  # for deepcopy
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np  
import numpy as np
import random
import math
from collections import Counter
import itertools
from pprint import pprint
import json
from datetime import timedelta, date
from dateutil import rrule
from datetime import datetime, timedelta
import datetime
import time
#from  random import *
import simpy
#from config import *
import networkx as nx
from itertools import islice
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from collections import Counter

from random import seed
from datetime import datetime
import networkx as nx
import math
from itertools import islice
import pandas as pd
import warnings
import time
import yaml
import pprint
import json
from functools import reduce
import operator



NO_OF_SLOT=[1,2,3,4]
RANDOM_SEED = [50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340]
MAX_TIME = 10000000
ERLANG_MIN = 400
ERLANG_MAX = 800
ERLANG_INC = 50  
NUM_OF_REQUESTS = 1000000
DEMAND_SLOT_QPSK=[1,2,3,4]
dem_slot=[2,3,4,5]
DEMAND_TYPE = [1,2,3,4]
TOPOLOGY = 'nkn'
HOLDING_TIME = 1.0
SLOTS = 360
SLOT_SIZE = 12.5
N_PATH = 1


topology = nx.read_weighted_edgelist('topology/' + TOPOLOGY, nodetype=int)
print(topology)


print(sum(dem_slot))
M=math.floor(SLOTS/sum(dem_slot))
print(M)
PARTITION_SIZES=[]
for i in range(len(dem_slot)):
  PARTITION_SIZES.append(M*dem_slot[i])
PARTITION_SIZES[3]=135
sum(PARTITION_SIZES)



#Deallocate spectrum after the expiary of holtding time
class Deallocate(object):
	def __init__(self, env):
		self.env = env
	def Run(self, count, path, spectrum, holding_time,partition):
		global topology
		yield self.env.timeout(holding_time)
		for i in range(0, (len(path)-1)):
			for slot in range(spectrum[0],spectrum[1]+1):
				topology[path[i]][path[i+1]][partition][slot] = 0


class Simulador(object):
	def __init__(self, env):
		self.env = env
		global topology
		global topology2
		demand_dict={}
		

		self.nodes = list(topology.nodes())
		self.random = random.Random()
		self.NumReqBlocked = 0 
		self.cont_req = 0
		self.k_paths = {}
		self.DemandT1=0
		self.DemandT2=0
		self.DemandT3=0
		self.DemandT4=0
		self.DemandT1_Blocked=0
		self.DemandT2_Blocked=0
		self.DemandT3_Blocked=0
		self.DemandT4_Blocked=0
		self.slot_usage_count={}
		self.LeftReq=0
		self.tot_holding_time=0
		self.all_neighboring_links={}
		self.link_load = {}

		for u, v in list(topology.edges):
			topology[u][v]['d1'] = [0] * PARTITION_SIZES[0]
			topology[u][v]['d2'] = [0] * PARTITION_SIZES[1]
			topology[u][v]['d3'] = [0] * PARTITION_SIZES[2]
			topology[u][v]['d4'] = [0] * PARTITION_SIZES[3]

			# topology2[u][v]['d1'] = [0] * PARTITION_SIZES[0]
			# topology2[u][v]['d2'] = [0] * PARTITION_SIZES[1]
			# topology2[u][v]['d3'] = [0] * PARTITION_SIZES[2]
			# topology2[u][v]['d4'] = [0] * PARTITION_SIZES[3]

			# self.slot_usage_count[topology2[u][v]['id']]=[0]*4

	def Run(self, rate):
		global topology
		global topology2

		for i in list(topology.nodes()):
			for j in list(topology.nodes()):
				if i!= j:
					self.k_paths[i,j] = self.k_shortest_paths(topology, i, j, N_PATH, weight='weight')

		#precompute all the neighboring links of all the paths in the network
		self.all_neighboring_links = self.find_neighboring_links_of_all_paths()

		for count in range(1, NUM_OF_REQUESTS + 1):
			yield self.env.timeout(self.random.expovariate(rate))
			src, dst = self.random.sample(self.nodes, 2)
			#bandwidth = self.random.choice(BANDWIDTH)
			holding_time = self.random.expovariate(HOLDING_TIME)
			num_slots=self.random.choice(DEMAND_SLOT_QPSK)
			demand_type=self.Demand_Type(num_slots)
			#self.Count_Demand_Type(demand_type)
			paths = self.k_paths[src,dst]
			#print("Demand Type: ",demand_type," Ht: ",holding_time," Path: ", paths)
			
			flag = 0
			for i in range(N_PATH):
				num_slots=self.Calculte_Num_Slot(demand_type)
				#print("Num Slots:",num_slots)
				free_block=self.PathIsAble_free_blocks(num_slots,paths[i],demand_type)
				#print("Path: ",paths[i])
				if free_block: # free_block has some items
					#print("Free Blocks: ",free_block)
					n_links = self.all_neighboring_links[tuple(paths[i])]
					occupied_block_on_neighborLink=self.check_common_occupied_blocks_block_key(free_block,n_links,demand_type)
					#print(occupied_block_on_neighborLink)
					if(occupied_block_on_neighborLink=={}):
						spectrum_block=[free_block[0][0],free_block[0][1]]
						self.cont_req += 1
						self.FirstFreeFit(count,spectrum_block[0],spectrum_block[1],paths[i],demand_type,holding_time)
						deallocate = Deallocate(self.env) #call Deaalocate Function
						self.env.process(deallocate.Run(count,paths[i],spectrum_block,holding_time,demand_type)) #start the deallocation process with timeout=holding_time Run(connectionID,path[10,3,1,2],[startSlotIndex,EndSlotIndex],holding_time)
						flag = 1 #connection established
						#print("Connection established on First free block ")
						break
					else:
						best_block=self.find_best_block_for_allocation(occupied_block_on_neighborLink,holding_time)
						spectrum_block=[best_block[0],best_block[1]]
						self.cont_req +=1
						self.FirstFreeFit(count,spectrum_block[0],spectrum_block[1],paths[i],demand_type,holding_time)
						deallocate=Deallocate(self.env) #call Deaalocate Function
						self.env.process(deallocate.Run(count,paths[i],spectrum_block,holding_time,demand_type))
						flag=1
						#print("Connection established on best block")
						break
			if flag == 0:
					# print("Connection Blocked")
					self.NumReqBlocked +=1
					self.Count_Demand_Type_Blocked(demand_type)


	def find_neighboring_links_of_a_path(self,path):
		global topology
		neighboring_links = set()
		path_edges = set(zip(path, path[1:]))  # Create a set of edges that are part of the path
		# Iterate through each node in the path
		for node_id in path:
			for neighbor in topology.neighbors(node_id):
				if (node_id, neighbor) not in path_edges and (neighbor, node_id) not in path_edges:
					neighboring_links.add((node_id, neighbor))
					neighboring_links.add((neighbor, node_id))
		# Remove any links that directly connect nodes in the path
		for edge in path_edges:
			neighboring_links.discard(edge)
			neighboring_links.discard((edge[1], edge[0]))  # Remove reverse direction if added
		unique_links = [tuple(sorted(set(link))) for link in neighboring_links]
		unique_links = list(set(unique_links))
		return unique_links

	def find_neighboring_links_of_all_paths(self):
		global topology
		neighborLinks = {}  # Dictionary to store neighboring links of all possible paths
		for path_key, path in self.k_paths.items():
			converted_list = [item for sublist in path for item in sublist]
			#print(converted_list)
			neighboring_links = self.find_neighboring_links_of_a_path(converted_list)
			neighborLinks[tuple(converted_list)] = neighboring_links  # Convert path_key to tuple
		return neighborLinks
	

		#find free blocks on selected path [0,2,3]
	#find free blocks on selected path [0,2,3]
	def PathIsAble_free_blocks(self, nslots, path, partition):
		global topology
		free_blocks = []  # List to store the start and end of each non-overlapping free block of size nslots
		slot = 0
		while slot < len(topology[path[0]][path[1]][partition]):
			if all(topology[path[ind]][path[ind + 1]][partition][slot] == 0 for ind in range(len(path) - 1)):
				block_start = slot
				while (slot < len(topology[path[0]][path[1]][partition]) and
					all(topology[path[ind]][path[ind + 1]][partition][slot] == 0 for ind in range(len(path) - 1))):
					slot += 1
				block_end = slot - 1  # slot is now one past the end of the free block
				block_length = block_end - block_start + 1
				if block_length >= nslots:
					free_blocks.append((block_start, block_start + nslots - 1))
			else:
				slot += 1  # Move to the next slot if the current slot is not free across the entire path
		return free_blocks

	
	def check_common_occupied_blocks_block_key(self, free_blocks, neighbor_links, partition):
		global topology
		occupied_blocks = {}

		for block_start, block_end in free_blocks:
			for link in neighbor_links:
				link_data = topology[link[0]][link[1]][partition]
				for slot in range(block_start, block_end + 1):
					release_time = link_data[slot]
					if release_time != 0:  # Slot is occupied
						release_time = release_time[1]
						block_key = (block_start, block_end)
						if block_key not in occupied_blocks:
							occupied_blocks[block_key] = {}
						if link not in occupied_blocks[block_key]:
							occupied_blocks[block_key][link] = []
						if release_time not in occupied_blocks[block_key][link]:
							occupied_blocks[block_key][link].append(release_time)

		sorted_occupied_blocks = dict(sorted(occupied_blocks.items(), key=lambda item: len(item[1]), reverse=True))
		return sorted_occupied_blocks

	

	def find_best_block_for_allocation(self, occupied_block_on_neighborLink, ht):
		current_r_rtime = time.time() + ht
		data = []

		for block_key, link_data in occupied_block_on_neighborLink.items():
			rt_list = [release_time for link_release_times in link_data.values() for release_time in link_release_times]
			rt_list.append(current_r_rtime)
			num_link = len(rt_list)
			variance = np.var(rt_list) if rt_list else 0
			data.append((block_key, rt_list, variance, num_link))

		variance_weight = 0.5
		link_count_weight = 0.5

		# Initialize variables to track the best list found
		best_block_key = None
		best_score = float('inf')  # Initialize with infinity to ensure first list is selected

		# Iterate through the data
		for block_key, rt_list, variance, link_count in data:
			score = variance_weight * variance + link_count_weight * link_count
			if score < best_score:
				best_block_key = block_key
				best_score = score

		return best_block_key
	

	 #Perform spectrum allocation on best block
	def FirstFreeFit(self,count,i,j,path,partition,ht): #FirstFit(connectionID, startSlotIndex, endSlotIndex, [10,3,1,2])
		global topology
		beginning = i #startSlotIndex
		end =j #endSlotIndex
		relese_time=time.time()+ht
		#print(relese_time)
		for i in range(0,len(path)-1): #for i in range(0, path_len-1) i.e for all the links in the path
			for slot in range(beginning,end): # for slot in (startSlotIndex,endSlotIndex)
				#print(slot)
				#topology[path[i]][path[i+1]]['capacity'][slot] = count #mark each slot with the connectionID
				topology[path[i]][path[i+1]][partition][slot] = (count,relese_time)
			#print(topology[path[i]][path[i+1]]['capacity'][slot])
			#topology[path[i]][path[i+1]]['capacity'][end] = ('GB',relese_time)#'GB' #mark last slot in each link with 'GB' GuardBand
		for k in range(len(path)-1):
			link = (min(path[k], path[k + 1]), max(path[k], path[k + 1]))
			if link in self.link_load:
				self.link_load[link] +=1
			else:
				self.link_load[link] = 1

  # Calculates path distance according to edge weights              
	def Distance(self, path):
		global topology
		global topology2 
		path_length = 0
		for i in range(0, (len(path)-1)):
			path_length += topology[path[i]][path[i+1]]['weight']
		return (path_length)

	#Calculates the k-shortest paths between o-d pairs
	def k_shortest_paths(self,G, source, target, k, weight='weight'):
		return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))
	
	def computeUsageTime(self):
		global topology2
		link=0
		for u, v in list(topology2.edges):
			xx=[]
			for dem in ['d1','d2','d3','d4']:
				print("Partition:",dem,sum(topology2[u][v][dem]))
				xx.append(sum(topology2[u][v][dem]))
			self.slot_usage_count[link]=xx
		
	#Identify the demand type based on the bandwidth
	def Demand_Type(self, bandwidth):
		if bandwidth==1:
			return 'd1'
		elif bandwidth==2:
			return 'd2'
		elif bandwidth==3:
			return 'd3'
		else:
			return 'd4'
		
	#compute no of slot required for each demand type using QPSk Modulation Scheme
	def Calculte_Num_Slot(self,demand):
		if demand=='d1':
			num_slots=1
		elif demand=='d2':
			num_slots=2
		elif demand=='d3':
			num_slots=3
		else:
			num_slots=4
		return num_slots

	#Count total no of requested demands for each type
	def Count_Demand_Type(self,demand):
		if demand=='d1':
			self.DemandT1+=1
		elif demand=='d2':
			self.DemandT2+=1
		elif demand=='d3':
			self.DemandT3+=1
		else:
			self.DemandT4+=1

	#Count total no of Blocked request for each demand type
	def Count_Demand_Type_Blocked(self,demand):
		if demand=='d1':
			self.DemandT1_Blocked+=1
		elif demand=='d2':
			self.DemandT2_Blocked+=1
		elif demand=='d3':
			self.DemandT3_Blocked+=1
		else:
			self.DemandT4_Blocked+=1

	#Perform spectrum allocation using First-fit
	def FirstFit(self,count,i,j,path,partition,holding_time):
		global topology
		global topology2
		beginning = i 
		end =j
		numslot=(end-beginning)+1
		duration=holding_time*numslot
		if partition=='d1':
			p=0
		elif partition=='d2':
			p=1
		elif partition=='d3':
			p=2
		else:
			p=3
		for i in range(0,len(path)-1):
			for slot in range(beginning,end):
				topology[path[i]][path[i+1]][partition][slot] = count
			topology[path[i]][path[i+1]][partition][end] = 'GB'
			#self.slot_usage_count[topology2[path[i]][path[i+1]]['id']][p]+=duration #duration= numslot*holding time

	# Checks if the chosen path has spectrum available for the requested demand
	def PathIsAble(self, nslots,path,partition):
		global topology
		global topology2
		cont = 0
		t = 0
		for slot in range (0,len(topology[path[0]][path[1]][partition])):
			if topology[path[0]][path[1]][partition][slot] == 0:
				k = 0
				for ind in range(0,len(path)-1):
					if topology[path[ind]][path[ind+1]][partition][slot] == 0:
						k += 1
				if k == len(path)-1:
					cont += 1
					if cont == 1:
						i = slot
					if cont > nslots:
						j = slot
						return [True,i,j]
					if slot == len(topology[path[0]][path[1]][partition])-1:
							return [False,0,0]
				else:
					cont = 0
					if slot == len(topology[path[0]][path[1]][partition])-1:
						return [False,0,0]
			else:
				cont = 0
				if slot == len(topology[path[0]][path[1]][partition])-1:
					return [False,0,0]




total_blocked={}
for e in range(ERLANG_MIN, ERLANG_MAX+1, ERLANG_INC):

  rep_block={}
  for rep in range(10):
    rate = e / HOLDING_TIME
    seed(RANDOM_SEED[rep])
    #print("Env created")
    env = simpy.Environment()
    simulador = Simulador(env)
    env.process(simulador.Run(rate))
    env.run()

    print("Rate: " ,rate,"Erlang", e, "Simulation...", rep)
    print("blocked", simulador.NumReqBlocked, "#Requests", NUM_OF_REQUESTS)
    print("Demand Type:",simulador.DemandT1,simulador.DemandT2,simulador.DemandT3,simulador.DemandT4)
    print("Demand Type Blocked:",simulador.DemandT1_Blocked,simulador.DemandT2_Blocked,simulador.DemandT3_Blocked,simulador.DemandT4_Blocked)
    print('-------------------------------------------------------------')
    rep_block[rep]=((simulador.NumReqBlocked/NUM_OF_REQUESTS))
    total_blocked[e]=rep_block

    #print(e,simulador.NumReqBlocked)



