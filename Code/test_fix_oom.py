
import unittest
import logging
from unittest.mock import MagicMock, patch
import networkx as nx
import sys
import os

# Ensure we can import the module
sys.path.append(os.getcwd())
import importlib
tp = importlib.import_module("05_tda_pipeline")

class TestTDAPipelineSafeGuard(unittest.TestCase):
    def test_large_graph_safeguard(self):
        print("\n[Test] Verifying Large Graph Visualization Safeguard...")
        
        # 1. Mock DataFrame
        mock_df = MagicMock()
        mock_df.columns = ['Total_Cost', 'Visit_Count', 'Comorbidity_Count']
        
        # 2. Initialize Pipeline
        pipeline = tp.TDAPipeline(mock_df, [], [], output_path="dummy.html")
        
        # 3. Create a Large Fake Graph
        # TDA graph format is usually dict(nodes={id: [indices]}, links={...})
        # But save_graph_as_png uses km.adapter.to_nx(self.graph)
        # So we can just mock pipeline.graph and patch km.adapter.to_nx
        
        pipeline.graph = {'nodes': {}, 'links': {}}
        
        # Mock km.adapter.to_nx to return a large NetworkX graph
        large_nx_graph = nx.Graph()
        large_nx_graph.add_nodes_from(range(6000)) # 6000 > 5000 threshold
        
        with patch('kmapper.adapter.to_nx', return_value=large_nx_graph) as mock_converter:
             with patch('matplotlib.pyplot.savefig') as mock_save:
                pipeline.save_graph_as_png("should_be_skipped.png")
                
                # Check that savefig was NOT called
                if mock_save.called:
                    self.fail("Matplotlib savefig() was called despite graph size > 5000!")
                else:
                    print("✅ check passed: savefig() was NOT called.")

    def test_small_graph_proceeds(self):
        print("\n[Test] Verifying Small Graph Normal Behavior...")
        
        # 1. Mock DataFrame
        mock_df = MagicMock()
        mock_df.to_pandas.return_value = mock_df # rudimentary mocking
        # Ensure columns check works
        mock_df.columns = ['Total_Cost']
        
        # 2. Pipeline
        pipeline = tp.TDAPipeline(mock_df, [], [], output_path="dummy.html")
        pipeline.df_pandas = mock_df # simplistic 
        pipeline.df_lens = mock_df
        
        # 3. Small Graph
        # Ensure node IDs match between pipeline.graph and the converted nx_graph
        pipeline.graph = {'nodes': {i: [0] for i in range(100)}, 'links': {}}
        small_nx_graph = nx.Graph()
        small_nx_graph.add_nodes_from(range(100))
        
        with patch('kmapper.adapter.to_nx', return_value=small_nx_graph):
            with patch('matplotlib.pyplot.savefig') as mock_save:
                # We expect it to try drawing and saving. 
                # Note: kamada_kawai_layout might fail on empty mock graph, so we mock layout too if needed.
                # But let's just check if it gets PAST the safeguard.
                with patch('networkx.kamada_kawai_layout', return_value={i: (0,0) for i in range(100)}):
                    with patch('networkx.draw_networkx_nodes'): 
                        with patch('networkx.draw_networkx_edges'):
                             pipeline.save_graph_as_png("should_run.png")
                             
                             if mock_save.called:
                                 print("✅ check passed: savefig() WAS called for small graph.")
                             else:
                                 self.fail("Matplotlib savefig() was NOT called for small graph!")

if __name__ == '__main__':
    unittest.main()
