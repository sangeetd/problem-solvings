/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.logging.Level;
import java.util.logging.Logger;

public class GraphTest {
    
    public static void main(String[] args) {
        
        /*try {
            GraphStructure g = new GraphStructure(5, GraphStructure.GRAPH_UNDIRECTED);
            
            g.addVertexConnections(0, 1);
            g.addVertexConnections(0, 2);
            g.addVertexConnections(1, 2);
            g.addVertexConnections(0, 3);
            g.addVertexConnections(3, 4);
            
            g.printAdjList();
            
//            g.graphBFS(0);
//            g.graphBFS(2);
            
//            g.ssspAsBFS(0, 3);
//            g.ssspAsBFS(4, 3);
            
//            g.graphDFS(0);
            
            g.ssspForDijkstra(0);
//            g.ssspForDijkstra(3);
            
        } catch (Exception ex) {
            Logger.getLogger(GraphTest.class.getName()).log(Level.SEVERE, null, ex);
        }*/
        
        /*try {
            GraphStructure g = new GraphStructure(5, GraphStructure.GRAPH_DIRECTED);
            
//            g.addVertexConnections(0, 1);
//            g.addVertexConnections(0, 2);
//            g.addVertexConnections(1, 2);
//            g.addVertexConnections(0, 3);
//            g.addVertexConnections(3, 4);
            
            g.addVertexConnections(0, 1, 0, 3);
            g.addVertexConnections(0, 2, 0, 1);
            g.addVertexConnections(1, 2, 0, 2);
            g.addVertexConnections(1, 3, 0, 5);
            g.addVertexConnections(2, 4, 0, 1);
            //g.addVertexConnections(3, 1, 0, 2);
            
            g.printAdjList();
            
//            g.graphBFS(0);
//            
//            g.graphDFS(0);
//            
//            g.ssspAsBFS(0);
//            
//            g.ssspForDijkstra(3);
            
            g.ssspForBellmanFord(0);
            
            
        } catch (Exception ex) {
            Logger.getLogger(GraphTest.class.getName()).log(Level.SEVERE, null, ex);
        }*/
        
        /*try {
            GraphStructure g = new GraphStructure(4, GraphStructure.GRAPH_DIRECTED);
            
            
            g.addVertexConnections(0, 3, 0, -10);
            g.addVertexConnections(3, 2, 0, 1);
            g.addVertexConnections(2, 1, 0, 2);
            g.addVertexConnections(1, 0, 0, 5);
            g.addVertexConnections(3, 1, 0, -6);
            //g.addVertexConnections(3, 1, 0, 2);
            
            g.printAdjList();
            
            g.ssspForBellmanFord(0);
            g.ssspForDijkstra(0);
            
            
        } catch (Exception ex) {
            Logger.getLogger(GraphTest.class.getName()).log(Level.SEVERE, null, ex);
        }*/
        
        /*try {
            GraphStructure g = new GraphStructure(4, GraphStructure.GRAPH_DIRECTED);
            
            
            g.addVertexConnections(0, 1, 0, 8);
            g.addVertexConnections(0, 3, 0, 1);
            g.addVertexConnections(1, 2, 0, 1);
            g.addVertexConnections(2, 0, 0, 4);
            g.addVertexConnections(3, 1, 0, 2);
            g.addVertexConnections(3, 2, 0, 9);
            
            g.printAdjList();
            
//            g.apspForBFS();
            
            g.apspForFloydWarshall();
//            g.kruskalForMCST();
            
        } catch (Exception ex) {
            Logger.getLogger(GraphTest.class.getName()).log(Level.SEVERE, null, ex);
        }*/
        
        
        try {
            GraphStructure g = new GraphStructure(5, GraphStructure.GRAPH_UNDIRECTED);
            
            
            g.addVertexConnections(0, 1, 15, 15);
            g.addVertexConnections(0, 2, 20, 20);
            g.addVertexConnections(1, 2, 13, 13);
            g.addVertexConnections(1, 3, 5, 5);
            g.addVertexConnections(2, 3, 10, 10);
            g.addVertexConnections(2, 4, 6, 6);
            g.addVertexConnections(3, 4, 8, 8);
            
//            g.addVertexConnections(0, 1, 10, 10);
//            g.addVertexConnections(0, 2, 20, 20);
//            g.addVertexConnections(1, 2, 30, 30);
//            g.addVertexConnections(1, 3, 25, 25);
//            g.addVertexConnections(2, 3, 15, 15);
//            g.addVertexConnections(2, 4, 6, 6);
//            g.addVertexConnections(3, 4, 8, 8);
            
            g.printAdjList();
            
            GraphStructure g2= g.kruskalForMCST();
            System.out.println("Minimum cost spanning tree obtained from graph g kruskal");
            g2.printAdjList();
//            g2.graphBFS(0);
//            g2.ssspAsBFS(0);
//            g2.ssspForDijkstra(0);
//            g2.ssspForBellmanFord(0);
//            g2.apspForFloydWarshall();
            
            GraphStructure g3= g.primsForMCST();
            System.out.println("Minimum cost spanning tree obtained from graph g prims");
            g3.printAdjList();
            
        } catch (Exception ex) {
            Logger.getLogger(GraphTest.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        
    }
    
}
