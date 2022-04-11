import numpy as np
import scipy.sparse as sparse

from collections import OrderedDict
from sima.ROI import ROI, ROIList

class CNMFData(object):
    def __init__(self, Y=None, Ypath=None, A=None, C=None, b=None, f=None, S=None, template=None, dims=None, patch_structure=None):
        self.dims = dims
        self.A = AROIs(matrix=A, dims=dims)
        self.C = C
        self.b = b
        self.f = f 
        self.S = S
        self.template=template
        self.Y = Y
        self.Ypath = Ypath

        if patch_structure is not None:
            self.slices = self.calculatePatches(patch_structure)

    def calculatePatches(self, patch_structure):
        slices = {}

        if type(patch_structure[0]) is int:
            assert len(patch_structure) == len(self.template.shape)
            self.patch_tuples = it.product(*[range(k) for k in patch_structure])
            grid = [np.linspace(0, n, k+1, dtype=int) for n,k in zip(self.shape, patch_structure)]

            for comb in self.patch_tuples:
                sls = []
                tups = []
                for dim,dimidx in enumerate(comb):
                    beg_end = (grid[dim][dimidx], grid[dim][dimidx+1])
                    tup.append(beg_end)
                    sl.append(slice(*beg_end))
                slices[tuple(tup)] = tuple(sl)
        elif type(structure[0]) is tuple:
            for tup in structure:
                slices[tup] = tuple([slice(t) for t in tup])

        return slices


class AROIs(object):
    def __init__(self, roilist=None, matrix=None, tensor=None, dims=None, hold_data=False,
                 label=None, rids=None, labels=None, tags=None):
        self._matrix = None
        self._tensor = None
        self._roilist = None
        self.label = label
                
        self.hold_data = hold_data

        # sima-specific parameters
        self.labels = labels
        self.tags = tags
        
        if matrix is not None:
            if dims is None:
                raise ValueError("Matrix form is ambiguous without dims")
            else:
                self.dim0, self.dim1, self.dim2 = dims
            
            _,self.nrois = matrix.shape
            
            if sparse.issparse(matrix):
                self._matrix = matrix
            else:
                self._matrix = sparse.csc_matrix(matrix)
                  
        elif tensor is not None:
            self.nrois,self.dim0,self.dim1,self.dim2 = tensor.shape
            self._tensor = tensor
            self._matrix = self.getMatrix()
            
        elif roilist is not None:
            self.nrois = len(roilist)
            self.dim0,self.dim1,self.dim2 = roilist[0].im_shape
            
            if rids is None:
                rids = map(lambda x: x.id, roilist)
            
            self._roilist = roilist
            self._matrix = self.getMatrix()
            
        else:
            raise ValueError("Must initialize with one of 'roilist', 'matrix', or 'tensor'")
        
        if rids is not None and len(rids) == len(self):
            self.rids = OrderedDict([(rid,i) for i,rid in enumerate(rids)])
        else:
            self.rids = OrderedDict([(i,i) for i in range(len(self))])
        
                    
    @property
    def matrix(self):
        return self.getMatrix()

    @property
    def dense(self):
        return self.getMatrix(sparsified=False)
    
    @property
    def tensor(self):
        return self.getTensor()
    
    @property
    def roilist(self):
        return self.getROIList()
    
    @property
    def dims(self):
        return self.dim0, self.dim1, self.dim2

    @property
    def shape(self):
        return self.dims

    def getMatrix(self, sparsified=True):
        if self._matrix is not None:
            return self._matrix if sparsified else np.asarray(self._matrix.todense())
        
        elif self._tensor is not None:
            nrois,d0,d1,d2 = self._tensor.shape
            
            assert (d0 == self.dim0) & (d2 == self.dim2) & (d1 == self.dim1)
            
            reshaped_array = self._tensor.transpose((1, 3, 2, 0))
            reshaped_array = reshaped_array.reshape((self.dim0*self.dim1*self.dim2, nrois),
                                                   order="C")
            
            self._matrix = sparse.csc_matrix(reshaped_array)
            
            if not self.hold_data:
                self._tensor = None
        
            return self._matrix if sparsified else reshaped_array
        
        elif self._roilist is not None:
            masks = [[np.asarray(plane.todense()) for plane in roi.mask] for roi in self._roilist]
            masks = [np.stack(mask,axis=0) for mask in masks]

            self.rids = {roi.id:i for i,roi in enumerate(self._roilist)}
            self.tags = {roi.id:roi.tags for roi in self._roilist}
            self.labels = {roi.id:roi.label for roi in self._roilist}
            
            # I am evil >:D
            self._tensor = np.stack(masks, axis=0)
            matrix = self.getMatrix(sparsified=sparsified)
            
            if not self.hold_data:
                self._roilist = None
                
            return matrix
    
    def getTensor(self, transpose=None):
        if self._tensor is not None:
            return self._tensor
        
        elif self._matrix is not None:
            npx, nrois = self._matrix.shape
            
            assert npx == self.dim0*self.dim1*self.dim2
            
            array = self.getMatrix(sparsified=False)
            
            reshaped_array = array.reshape((self.dim0,self.dim2,self.dim1,nrois),
                                          order="C")
            # New order: roi x z x X x y
            reshaped_array = reshaped_array.transpose((3,0,2,1))
            
            if transpose is not None:
                reshaped_array = reshaped_array.transpose(transpose)
            
            if self.hold_data:
                self._tensor = reshaped_array
        
            return reshaped_array
        
        else:
            raise AssertionError("I don't know how to make a tensor. This should never happen")
        
    def getROIList(self,expt=None,as_bool=False,bin_threshold=0.):
        if self._roilist is not None:
            return self._roilist
        elif self._matrix is not None:
            roilist = []
                        
            As = self.getMatrix(sparsified=False)

            if expt is not None:
                starttime = expt.get("startTime")
            else:
                starttime=None
            
            for rid, idx in self.rids.items():
                A = As[:,idx]
                dim0,dim1,dim2 = self.dims
                if rid == idx:
                    roi_id = "{}_{}".format(starttime,
                                        rid)
                else:
                    roi_id = rid
                mask = A.reshape((dim0,dim2,dim1)).transpose((0,2,1))
                if as_bool:
                    mask = mask > bin_threshold

                if hasattr(self, "labels") and self.labels is not None and rid in self.labels:
                    label = self.labels[rid]
                else:
                    label = self.label

                if hasattr(self, "tags") and self.tags is not None and rid in self.tags:
                    tags = self.tags[rid]
                else:
                    tags = []

                roi = ROI(mask=mask, id=roi_id, label=label, tags=tags)
                roilist.append(roi)
            roilist = ROIList(roilist)
           # roilist = sorted(roilist, key=lambda x: x.id)

            unique, cts = np.unique(self.rids, return_counts=True)
            if not np.all(cts==1):
                for rid,num in zip(unique[cts!=1],cts[cts!=1]):
                    print("WARNING: {} occurs {} times".format(rid,num))
                    
            if self.hold_data:
                self._roilist = roilist
            # For now; eventually just save
            return roilist
        else:
            raise AssertionError("I don't know how to make an roilist. This should never happen")
        
    def keys(self):
        return self.rids.keys()

    def to_patches(self, patches, nonzero=False):
        for patch in patches:

            rois_patch = self.tensor[(slice(None),) + patch]
            rids = self.rids.keys()

            if nonzero is True:
                nonzero_roi_idxes, = np.nonzero(rois_patch.sum(axis=tuple(np.arange(1,len(self.dims)+1))))
                rids = [rids[i] for i in nonzero_roi_idxes]
                rois_patch = rois_patch[nonzero_roi_idxes]

            yield AROIs(tensor=rois_patch, rids=rids)

    def __iter__(self):
        for i, rid in enumerate(self.rids.keys()):
            yield AROIs(matrix=self.matrix[:, i], 
                         dims=self.dims, rids=[rid])

    def __len__(self):
        return self.nrois
    
    def __getitem__(self, key):
        if type(key) is slice or (type(key) is tuple and type(key[0]) is slice):
            return AROIs(matrix=self.matrix[key])

        elif type(key) is tuple:
            k,fmt = key
            
            idx = self.rids[k]
            
            if fmt == 'tensor':
                return self.tensor[idx,:,:,:]
            elif fmt == 'ROI':
                return self.roilist[idx]
            elif fmt == 'dense':
                return self.getMatrix(sparsified=False)[:,idx]
            else:
                return self[k]
            
        elif type(key) is list:
            return [self[k] for k in key]
                
        else:        
            idx = self.rids[key]
            return AROIs(matrix=self.matrix[:,idx], dims=self.dims)
