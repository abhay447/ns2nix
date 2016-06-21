#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import h5py
import nix
import neuroshare as ns
import numpy as np
import getopt
import time

class ProgressIndicator(object):
	def __init__(self, offset=0):
		self._cur_value = offset
		self._max_value = 0

	def setup(self, max_value):
		self._max_value = max_value
		self.progress(self._max_value, 0)

	def __add__(self, other):
		self._cur_value += other
		self.progress(self._max_value, self._cur_value)
		return self

	def progress(self, max_value, cur_value):
		pass


class Converter(object):
	def __init__(self, filepath, output=None, progress=None):
		if not output:
			(basefile, ext) = os.path.splitext(filepath)
			#output = "%s.hdf5" % basefile
			output = "%s.h5" % basefile

		nf = ns.File(filepath)
		#h5 = h5py.File(output, 'w')
		nixF = nix.File.open(output,nix.FileMode.Overwrite)
		nixB = nixF.create_block("main_block","nix.session")

		self._nf = nf
		#self._h5 = h5
		self._nixF = nixF
		self._nixB = nixB
		self._groups = {}
		self.convert_map = {1: self.convert_event,
							2: self.convert_analog,
							3: self.convert_segment,
							4: self.convert_neural}
		if not progress:
			progress = ProgressIndicator()
		self._progress = progress

	def get_group_for_type(self, entity_type):
		name_map = {1: 'Event',
					2: 'Analog',
					3: 'Segment',
					4: 'Neural'}

		if entity_type not in self._groups:
			name = name_map[entity_type]
			#group = self._h5.create_group(name)
			group = self._nixB.create_group(name,"group")
			#self._groups[entity_type] = group
			self._groups[entity_type] = group

		#return self._groups[entity_type]
		return self._groups[entity_type]

	def convert(self):
		progress = self._progress
		progress.setup(len(self._nf.entities))
		#self.copy_metadata(self._h5, self._nf.metadata_raw)
		self.copy_metadata(self._nixF,self._nixB, self._nf.metadata_raw)
		for entity in self._nf.entities:
			conv = self.convert_map[entity.entity_type]
			conv(entity)
			progress + 1

		#self._h5.close()
		self._nixF.close()

	def convert_event(self, event):
		dtype = self.dtype_by_event(event)
		nitems = event.item_count
		data = np.empty([nitems], dtype)
		for n in range(0, event.item_count):
			data[n] = event.get_data(n)

		group = self.get_group_for_type(event.entity_type)
		dset = self._nixB.create_data_array(event.label, "nix.event", data=data)
		group.data_arrays.append(dset)
		#dset = group.create_dataset(event.label, data=data)
		self.copy_metadata(self._nixF,dset, event.metadata_raw)

	def convert_analog(self, analog):
		(data, times, ic) = analog.get_data()
		group = self.get_group_for_type(analog.entity_type)
		dset = self._nixB.create_data_array(analog.label, "nix.analog", dtype=np.double, data=data.T)
		dim = dset.append_range_dimension(times)
		dim.unit = "s"
		dim.label = "time"
		group.data_arrays.append(dset)
		self.copy_metadata(self._nixF,dset, analog.metadata_raw)

	def convert_segment(self, segment):
		if not segment.item_count:
			return
		seg_group = self.get_group_for_type(segment.entity_type)
		self.copy_metadata(self._nixF,seg_group, segment.metadata_raw, prefix=str(segment.id)+"_")
		
		for index in range(0, segment.source_count):
			source = segment.sources[index]
			name = 'SourceInfo.%d.' % index
			#self.copy_metadata(seg_group, source.metadata_raw, prefix=name)
			self.copy_metadata(self._nixF,seg_group, source.metadata_raw, prefix=str(segment.id)+"_"+name)

		for index in range(0, segment.item_count):
			(data, timestamp, samples, unit) = segment.get_data(index)
			name = '%d - %f' % (index, timestamp)
			dset = self._nixB.create_data_array(str(segment.id)+"_"+name, "nix.segment", dtype=np.double, data=data.T)
			stepSize = 1/segment.metadata_raw['SampleRate']
			dim = dset.append_sampled_dimension(stepSize)
			dim.unit = 's'
			dim.offset = timestamp
			dim.label = "time"
			sec = self._nixF.create_section(str(segment.id)+"_"+name+"_metadata","odml.unit")
			sec.create_property('unitID',nix.Value(unit))
			dset.metadata = sec
			seg_group.data_arrays.append(dset)

	def convert_neural(self, neural):
		data = neural.get_data()
		#group = self._groups[neural.entity_type]
		group = self.get_group_for_type(neural.entity_type)
		name = "%d - %s" % (neural.id, neural.label)
		dset = self._nixB.create_data_array(name, "nix.neural", dtype=np.double, data=data)
		group.data_arrays.append(dset)
		self.copy_metadata(self._nixF,dset, neural.metadata_raw)

	@classmethod
	def copy_metadata(cls,mainFile, target, metadata, prefix=None):
		try:
			name = target.name
		except:
			name = "File"
		if prefix is not None:
			name = prefix+name
		sec = mainFile.create_section(name+' metadata','odml.recording')
		for (key, value) in metadata.items():
			if prefix is not None:
				key = prefix + key
			sec.create_property(key,nix.Value(value))
		target.metadata = sec

	@classmethod
	def dtype_by_event(cls, event):
		type_map = {ns.EventEntity.EVENT_TEXT  : 'a',
					ns.EventEntity.EVENT_CSV   : 'a',
					ns.EventEntity.EVENT_BYTE  : 'b',
					ns.EventEntity.EVENT_WORD  : 'h',
					ns.EventEntity.EVENT_DWORD : 'i'}
		val_type = type_map[event.entity_type]
		if val_type == 'a':
			val_type += str(event.max_data_length)
		return np.dtype([('timestamp', 'd'), ('value', val_type)])


class ConsoleIndicator(ProgressIndicator):
	def __init__(self):
		super(ConsoleIndicator, self).__init__()
		self._size = 60
		self._last_msg = ""

	def progress(self,  max_value, cur_value):
		size = self._size
		prefix = "Converting"
		x = int(size*cur_value/max_value)
		msg = "%s [%s%s] %i/%i\r" % (prefix, "#"*x, "." * (size-x),
									 cur_value, max_value)
		self._last_msg = msg
		sys.stdout.write(msg)
		sys.stdout.flush()

	def cleanup(self):
		sys.stdout.write('%s\r' % (' '*len(self._last_msg)))
		sys.stdout.flush()


def main():
	opts, rem = getopt.getopt(sys.argv[1:], 'o:', ['output=',
												   'version=',
												   ])
	output = None
	for opt, arg in opts:
		if opt in ("-o", "--output"):
			output = arg

	if len(rem) != 1:
		print("Wrong number of arguments")
		return -1

	filename = rem[0]
	ci = ConsoleIndicator()
	converter = Converter(filename, output, progress=ci)
	start = time.time()
	converter.convert()
	ci.cleanup()
	print(time.time()-start)
	return 0


if __name__ == "__main__":
	res = main()
	sys.exit(res)
