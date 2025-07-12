import sys
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

class UnionFind:
    '''
    An implementation of a Union--Find class. The class performs path
    compression by default. It uses integers for storing one disjoint
    set, assuming that vertices are zero-indexed.
    '''

    def __init__(self, n_vertices):
        '''
        Initializes an empty Union--Find data structure for a given
        number of vertices.
        '''
        self._parent = np.arange(n_vertices, dtype=int)

    def find(self, u):
        '''
        Finds and returns the parent of u with respect to the hierarchy.
        '''
        if self._parent[u] == u:
            return u
        else:
            # Perform path collapse operation
            self._parent[u] = self.find(self._parent[u])
            return self._parent[u]

    def merge(self, u, v):
        '''
        Merges vertex u into the component of vertex v. Note the
        asymmetry of this operation.
        '''
        if u != v:
            self._parent[self.find(u)] = self.find(v)

    def roots(self):
        '''
        Generator expression for returning roots, i.e. components that
        are their own parents.
        '''
        for vertex, parent in enumerate(self._parent):
            if vertex == parent:
                yield vertex

class PersistentHomologyCalculation:
    def __call__(self, matrix):
        n_vertices = matrix.shape[0]
        uf = UnionFind(n_vertices)

        triu_indices = np.triu_indices_from(matrix)
        edge_weights = matrix[triu_indices]
        edge_indices = np.argsort(edge_weights, kind='stable')

        persistence_pairs = []

        for edge_index in edge_indices:
            edge_weight = edge_weights[edge_index]
            u = triu_indices[0][edge_index]
            v = triu_indices[1][edge_index]

            younger = uf.find(u)
            older = uf.find(v)

            if younger == older:
                continue
            elif younger > older:
                uf.merge(v, u)
            else:
                uf.merge(u, v)

            persistence_pairs.append((min(u, v), max(u, v)))

        return np.array(persistence_pairs), np.array([])


class TopologicalSignatureDistance(nn.Module):
    """Topological signature distance calculation"""
    def __init__(self, use_cycles=False, match_edges=None):
        super().__init__()
        self.use_cycles = use_cycles
        self.match_edges = match_edges
        self.ph_calculator = PersistentHomologyCalculation()

    def _get_pairings(self, distance_matrix):
        """Compute persistence pairs for distance matrix"""
        if isinstance(distance_matrix, torch.Tensor):
            distance_matrix = distance_matrix.detach().cpu().numpy()
        return self.ph_calculator(distance_matrix)

    def _select_distances(self, distance_matrix, pairs):
        """Select distances from persistence pairs"""
        pairs_0, pairs_1 = pairs
        
        # Handle 0D features
        if len(pairs_0) > 0:
            selected = distance_matrix[pairs_0[:, 0], pairs_0[:, 1]]
        else:
            selected = torch.tensor([], device=distance_matrix.device)
        
        # Handle 1D features (not implemented)
        if self.use_cycles and len(pairs_1) > 0:
            # Placeholder for cycle features
            pass
            
        return selected

    @staticmethod
    def sig_error(sig1, sig2):
        """Compute signature error"""
        return ((sig1 - sig2)**2).sum()

    def forward(self, dist1, dist2):
        """Compute topological distance between two distance matrices"""
        pairs1 = self._get_pairings(dist1)
        pairs2 = self._get_pairings(dist2)
        
        # Calculate matched pairs metric
        matched_pairs = self._count_matching_pairs(pairs1[0], pairs2[0])
        metrics = {'matched_pairs_0D': matched_pairs}
        
        # Compute topological distance
        if self.match_edges == 'symmetric':
            sig1_1 = self._select_distances(dist1, pairs1)
            sig1_2 = self._select_distances(dist2, pairs1)
            
            sig2_2 = self._select_distances(dist2, pairs2)
            sig2_1 = self._select_distances(dist1, pairs2)
            
            dist1_2 = self.sig_error(sig1_1, sig1_2)
            dist2_1 = self.sig_error(sig2_2, sig2_1)
            
            total_dist = dist1_2 + dist2_1
            metrics.update({
                'distance1-2': dist1_2.item(),
                'distance2-1': dist2_1.item()
            })
        else:
            sig1 = self._select_distances(dist1, pairs1)
            sig2 = self._select_distances(dist2, pairs2)
            total_dist = self.sig_error(sig1, sig2)
        
        return total_dist, metrics

    @staticmethod
    def _count_matching_pairs(pairs1, pairs2):
        """Count matching persistence pairs"""
        if len(pairs1) == 0 or len(pairs2) == 0:
            return 0
        set1 = {tuple(pair) for pair in pairs1}
        set2 = {tuple(pair) for pair in pairs2}
        return len(set1.intersection(set2))
 