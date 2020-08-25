/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;

public class GraphStructure {

    private class Vertex {

        private Map<Integer, Integer> v;

        public Vertex(Map<Integer, Integer> v) {
            this.v = v;
        }

        public Map<Integer, Integer> getV() {
            return v;
        }

        public void setV(Map<Integer, Integer> v) {
            this.v = v;
        }

    }

    private List<Vertex> adjList;

    private int V;

    public static final int GRAPH_DIRECTED = 0;
    public static final int GRAPH_UNDIRECTED = 1;

    private int request;

    public GraphStructure(int V, int request) {
        this.V = V;
        this.request = request;

        adjList = new ArrayList<>();

        for (int i = 0; i < V; i++) {

            adjList.add(new Vertex(new HashMap<>()));

        }

    }

    public void addVertexConnections(int v1, int v2) throws Exception {

        addVertexConnections(v1, v2, 1, 1);

    }

    public void addVertexConnections(int v1, int v2, int costV1, int costV2) throws Exception {

        switch (this.request) {
            case GRAPH_DIRECTED:

                Vertex v = this.adjList.get(v1);
                Map<Integer, Integer> m = v.getV();
                m.put(v2, costV2);

                break;

            case GRAPH_UNDIRECTED:

                if (costV1 != costV2) {
                    throw new Exception("Undirected graph should have same cost values");
                }

                Vertex ver1 = this.adjList.get(v1);
                Map<Integer, Integer> m1 = ver1.getV();
                m1.put(v2, costV2);

                Vertex ver2 = this.adjList.get(v2);
                Map<Integer, Integer> m2 = ver2.getV();
                m2.put(v1, costV1);

                break;

            default: //none
        }

    }

    public void printAdjList() {

        for (int i = 0; i < this.V; i++) {

            System.out.println("\nvertex " + i + " links with");

            Vertex v = this.adjList.get(i);
            Map<Integer, Integer> m = v.getV();
            for (Map.Entry<Integer, Integer> e : m.entrySet()) {
                System.out.print(e.getKey() + "(" + e.getValue() + ") ");
            }

        }

    }

    public void graphBFS(int source) {

        System.out.println();

        boolean[] visited = new boolean[this.V];

        Queue<Integer> q = new LinkedList<>();
        q.add(source);

        while (!q.isEmpty()) {

            int u = q.poll();
            if (visited[u] == false) {
                visited[u] = true;

                //add connected vertes to queue
                Vertex v = this.adjList.get(u);
                Map<Integer, Integer> m = v.getV();
                for (Map.Entry<Integer, Integer> e : m.entrySet()) {
                    q.add(e.getKey());
                    //System.out.print(e.getKey() + "(" + e.getValue() + ") ");
                }
                System.out.print(u);
            }

        }

    }

    public void graphDFS(int source) {

        System.out.println();

        boolean[] visited = new boolean[this.V];

        Stack<Integer> s = new Stack<>();

        s.add(source);
        while (!s.isEmpty()) {

            int u = s.pop();

            if (visited[u] == false) {
                visited[u] = true;

                //add connected vertes to queue
                Vertex v = this.adjList.get(u);
                Map<Integer, Integer> m = v.getV();
                for (Map.Entry<Integer, Integer> e : m.entrySet()) {
                    s.add(e.getKey());
                    //System.out.print(e.getKey() + "(" + e.getValue() + ") ");
                }
                System.out.print(u);

            }

        }

    }

    public void ssspAsBFS(int source) {

        System.out.println();

        System.out.println("From source : " + source);

        boolean[] visited = new boolean[this.V];
        int[] parent = new int[this.V];
        parent[source] = -1;

        Queue<Integer> q = new LinkedList<>();
        q.add(source);

        while (!q.isEmpty()) {

            int u = q.poll();
            if (visited[u] == false) {
                visited[u] = true;

                //add connected vertes to queue
                Vertex v = this.adjList.get(u);
                Map<Integer, Integer> m = v.getV();
                for (Map.Entry<Integer, Integer> e : m.entrySet()) {
                    q.add(e.getKey());

                    if (visited[e.getKey()] == false) {
                        parent[e.getKey()] = u;
                    }

                }
            }

        }

        for (int i = 0; i < this.V; i++) {

            System.out.println(i + " parent is " + parent[i]);

        }

    }

    public void ssspForDFS(int source) {
        System.out.println("SSSP for DFS cant be possible as it goes deep down the graph");
    }

    public void ssspForDijkstra(int source) {

        System.out.println();

        System.out.println("From source : " + source);

        boolean[] visited = new boolean[this.V];
        int[] parent = new int[this.V];
        int[] dist = new int[this.V];

        for (int i = 0; i < this.V; i++) {
            dist[i] = Integer.MAX_VALUE;
        }

        parent[source] = -1;
        dist[source] = 0;

        Queue<Integer> q = new LinkedList<>();
        q.add(source);

        while (!q.isEmpty()) {

            int u = q.poll();

            if (visited[u] == false) {

                visited[u] = true;

                //get all connected vertex
                Vertex v = this.adjList.get(u);
                Map<Integer, Integer> m = v.getV();
                for (Map.Entry<Integer, Integer> e : m.entrySet()) {

                    q.add(e.getKey());

                    //calculte dist
                    int actualVerCost = e.getValue();
                    int uVerCost = dist[u];

                    if (uVerCost + actualVerCost <= dist[e.getKey()]
                            && visited[e.getKey()] == false
                            && dist[u] != Integer.MAX_VALUE) {

                        dist[e.getKey()] = uVerCost + actualVerCost;
                        parent[e.getKey()] = u;
                    }

                }

            }

        }

        for (int i = 0; i < this.V; i++) {

            System.out.println(i + " parent is " + parent[i]);

        }

        for (int i = 0; i < this.V; i++) {

            System.out.println(i + " vertex cost " + dist[i]);

        }

    }

    public void ssspForBellmanFord(int source) {

        System.out.println();

        System.out.println("From source : " + source);

        int[] dist = new int[this.V];

        for (int i = 0; i < this.V; i++) {
            dist[i] = Integer.MAX_VALUE;
        }

        dist[source] = 0;

        //loop -> V-1
        //loop all vertex
        //relaxtion of cost/dist optimising
        //loop itr at Vth to report for negative cycle or not
        int itr = 0;
        // 0->V-1 | 0 -> 5-1 | 0 -> 4 
        while (itr < this.V - 1) {

            for (int i = 0; i < this.V; i++) {
                //i means v1
                int u = i;

                Vertex v = this.adjList.get(u);
                Map<Integer, Integer> m = v.getV();
                for (Map.Entry<Integer, Integer> e : m.entrySet()) {

                    int actuaVerCost = e.getValue();
                    int uVerCost = dist[u];

                    if ((actuaVerCost + uVerCost) <= dist[e.getKey()] && dist[u] != Integer.MAX_VALUE) {
                        dist[e.getKey()] = actuaVerCost + uVerCost;
                    }

                }

            }

            System.out.println(itr + " iteration: ");

            for (int i = 0; i < this.V; i++) {

                System.out.println(i + " vertex cost " + dist[i]);

            }

            itr++;

        }

        System.out.println(" Vth iteration for negative cycle reporting: ");
        //just one iteration to analyse if dist[] is changing in this one or not
        boolean negativeCycleFound = false;
        for (int i = 0; i < this.V; i++) {
            //i means v1
            int u = i;

            Vertex v = this.adjList.get(u);
            Map<Integer, Integer> m = v.getV();
            for (Map.Entry<Integer, Integer> e : m.entrySet()) {

                //e.getKey() means v2
                //edge = v1,v2
                int actuaVerCost = e.getValue(); //v2(Cost)
                int uVerCost = dist[u]; // v1(Cost) optimised
                //System.out.println(dist[e.getKey()]+"  "+(actuaVerCost + uVerCost));
                if (dist[e.getKey()] > (actuaVerCost + uVerCost)) {
                    negativeCycleFound = true;
                }
            }

        }

        if (negativeCycleFound) {
            System.out.println("graph has negative cycle issue, Single source shortest path can never be found");
        } else {
            System.out.println("graph is OK");
        }

    }

    public void apspForBFS() {

        for (int v = 0; v < this.V; v++) {

            ssspAsBFS(v);

        }

    }

    public void apspForDijkstra() {

        for (int v = 0; v < this.V; v++) {

            ssspForDijkstra(v);

        }

    }

    public void apspForBellmanFord() {

        for (int v = 0; v < this.V; v++) {

            ssspForBellmanFord(v);

        }

    }

    public int[][] floydWarshallSingleSource(int via, int[][] dist) {

        System.out.println();

        System.out.println("via vertex : " + via);

//        int[] parent = new int[this.V];
//        parent[source] = -1;
        for (int u = 0; u < this.V; u++) {

            for (int v = 0; v < this.V; v++) {

                if ((dist[u][via] + dist[via][v]) < dist[u][v]
                        && u != v
                        && dist[u][via] != Integer.MAX_VALUE
                        && dist[via][v] != Integer.MAX_VALUE) {

                    //System.out.println((long)dist[1][0]+(long)dist[0][2]);
                    //System.out.println(u + "|" + via + "   " + via + "|" + v + "   " + u + "|" + v + "   " + (dist[u][via] + dist[via][v]) + "   " + dist[u][v]);
                    dist[u][v] = dist[u][via] + dist[via][v];
                }

            }

        }

        return dist;

    }

    public void apspForFloydWarshall() {

        int[][] dist = new int[this.V][this.V];

        for (int u = 0; u < this.V; u++) {

            for (int v = 0; v < this.V; v++) {

                dist[u][v] = Integer.MAX_VALUE;

                if (u == v) {
                    dist[u][v] = 0;
                }

            }

            Vertex ver = this.adjList.get(u);
            Map<Integer, Integer> m = ver.getV();
            for (Map.Entry<Integer, Integer> e : m.entrySet()) {
                dist[u][e.getKey()] = e.getValue();
            }

        }

        System.out.println("initial dist matrix");
        for (int u = 0; u < this.V; u++) {
            System.out.println();
            for (int v = 0; v < this.V; v++) {
                System.out.print("(" + u + "," + v + ")" + dist[u][v] + "  ");
            }

        }

        for (int v = 0; v < this.V; v++) {

            dist = floydWarshallSingleSource(v, dist);

            System.out.println("dist matrix updation ");
            for (int u = 0; u < this.V; u++) {
                System.out.println();
                for (int j = 0; j < this.V; j++) {
                    System.out.print("(" + u + "," + j + ")" + dist[u][j] + "  ");
                }

            }

        }

    }

    private List<Integer[]> incrOrderOfEdgeList() {

        List<Integer[]> l;
        //treemap keeps that key-value in sorted order of key
        //key-value = cost-list(edge)
        //edge = Int[] at 0->u, 1->v
        Map<Integer, List<Integer[]>> tm = new TreeMap<>();

        for (int u = 0; u < this.V; u++) {

            Vertex ver = this.adjList.get(u);
            Map<Integer, Integer> m = ver.getV();
            for (Map.Entry<Integer, Integer> e : m.entrySet()) {
                int v = e.getKey();
                int c = e.getValue();

                if (tm.containsKey(c)) {
                    List<Integer[]> l_ = tm.get(c);
                    l_.add(new Integer[]{u, v});
                    tm.put(c, l_);
                } else {
                    l = new ArrayList<>();
                    l.add(new Integer[]{u, v});
                    tm.put(c, l);
                }

            }

        }

        List<Integer[]> incOrderedEdgeList = new LinkedList<>();
        //putting all list(edge) in treemap to a single linkedlist(edgeAndCost) to maintain 
        //insertion order of edge
        //edgeAndCost = Int[] 0->u, 1->v, 2->c
        tm.entrySet().stream().forEach((e) -> {
            int c = e.getKey();
            List<Integer[]> ed = e.getValue();
            //System.out.println(c + "   ");
            ed.stream().forEach((uv) -> {
                //System.out.println(uv[0] + ", " + uv[1]);
                incOrderedEdgeList.add(new Integer[]{uv[0], uv[1], c});
            });
        });

        return incOrderedEdgeList;

    }

    public GraphStructure kruskalForMCST() throws Exception {

        List<Integer[]> incOrderedEdgeList = incrOrderOfEdgeList();

//        ordered edge list in insertion order also
//        for (Integer[] uvc : incOrderedEdgeList) {
//            System.out.println(uvc[0] + ", " + uvc[1]+"  --  "+uvc[2]);
//        }
        //Acc to kruskal algorithm
        //visit each edge in incr order of their edge-cost
        //put edges in set manner
        //if at point of time u,v is already present in the set it will create a cycle 
        //so ignore such kind of u,v and move to next incr edge and do same
        //for MST Edge connection is V-1 and min in edge-cost also
        List<Set<Integer>> listSet = new ArrayList<>();
        //listSet maintains all set inside the kruskal algo
        GraphStructure g = new GraphStructure(this.V, GraphStructure.GRAPH_UNDIRECTED);
        //creating new graph out of given graph as 
        //min spanning tree is subset of main graph
        int mcstCost = 0;
        for (Integer[] edgeAndCost : incOrderedEdgeList) {

            mcstCost += findAndUnion(listSet, edgeAndCost, g);
        }

        System.out.println("\nMCST from kruskal's algo: " + mcstCost);
        return g;

    }

    private int findAndUnion(List<Set<Integer>> listSet, Integer[] edgeAndCost, GraphStructure g) throws Exception {

//        testing /debugging prints
//        System.out.println("starting---"+u+","+v);
//        for (Set<Integer> sd : listSet) {
//                    System.out.println(sd.toString());
//        }
        int cost = 0;
        int u = edgeAndCost[0];
        int v = edgeAndCost[1];
        int c = edgeAndCost[2];

        if (listSet.isEmpty()) {
            //for first edge with min cost
            Set<Integer> s = new LinkedHashSet<>();
            s.add(u);
            s.add(v);
            listSet.add(s);
            cost += c;
            g.addVertexConnections(u, v, c, c);

        } else {
            int firstSetIndex = -1;
            int secondSetIndex = -1;
            boolean isCycle = false;
            for (int i = 0; i < listSet.size(); i++) {
                Set<Integer> s = listSet.get(i);

                if (s.contains(u) && s.contains(v)) {
                    //if any u,v is already a part of any set
                    //it will form cycle
                    //cycle case ignoe
                    isCycle = true;
                } else if (s.contains(u)) {
                    //it is for edge to edge connection
                    //set  = 0-3
                    //u-v = 3-4
                    //set extends = 0-3-4
                    firstSetIndex = i;
                } else if (s.contains(v)) {
                    //it is for edge to edge connection
                    //set  = 0-3
                    //u-v = 4-3
                    //set extends = 0-3-4
                    secondSetIndex = i;
                }

                if ((firstSetIndex != -1 && secondSetIndex == -1)
                        || (secondSetIndex != -1 && firstSetIndex == -1)) {
                    //if any u or v matches with a set in edge to edge
                    //this will add u or v to that set
                    s.add(u);
                    s.add(v);
                    cost += c;
                    g.addVertexConnections(u, v, c, c);
                } else if (firstSetIndex != secondSetIndex && firstSetIndex != -1 && secondSetIndex != -1) {
                    //if any u,v exist in two diff set
                    //firstIndex = set1 = 0-3
                    //secomdIndex = set2 = 1-2
                    //u,v = 3-1
                    //u in set1 & v in set2
                    //then do set1 union set2
                    //set = 0-3-2-1
                    union(listSet, firstSetIndex, secondSetIndex);
                }

            }

            if (!isCycle
                    && !((firstSetIndex != -1 && secondSetIndex == -1) || (secondSetIndex != -1 && firstSetIndex == -1))
                    && !(firstSetIndex != secondSetIndex && firstSetIndex != -1 && secondSetIndex != -1)) {

                //u,v is not creating cycle cond in any of the set
                //u,v is not involved in edge to edge in any of the set
                //u,v is not respectv. in any of the set that cause these sets to go in union
                Set<Integer> s = new LinkedHashSet<>();
                s.add(u);
                s.add(v);
                listSet.add(s);
                cost += c;
                g.addVertexConnections(u, v, c, c);
            }

        }
//        testing /debugging prints
//        System.out.println("endinf---");
//        for (Set<Integer> s : listSet) {
//            System.out.println(s.toString());
//        }

//        System.out.println("MCST sub graph---");
//        g.printAdjList();
        return cost;
    }

    private void union(List<Set<Integer>> listSet, int i, int j) {

        //#listSet is maintained as object ref -copy in parameter
        Set<Integer> s1 = listSet.get(i);
        Set<Integer> s2 = listSet.get(j);

        //less in size means less no of vertex item in set
        //for union items from small set is moved to other set
        //then that smaller set is removed from list of sets
        //implementation of set is needed because it ensures the unique 
        //occurence of any vertex
        if (s1.size() >= s2.size()) {
            s2.stream().forEach(x -> s1.add(x));
            listSet.remove(j);
        } else {
            s1.stream().forEach(x -> s2.add(x));
            listSet.remove(i);
        }

    }

    public GraphStructure primsForMCST() throws Exception {

        //Acc to prims algo
        //choose a least cost edge and maintain a set for that
        //now the further edges that are to be choosen should have least cost and also
        //that edge must form edge to edge connection with the vertex in the set.
        //rather than having a min cost with an edge sometimes we will have to choose that 
        //edge only that connects vertex to vertex in the set
        GraphStructure g = new GraphStructure(this.V, GraphStructure.GRAPH_UNDIRECTED);

        //inc order of edgeAndCost list
        List<Integer[]> incrOrderOfEdgeList = incrOrderOfEdgeList();

        //set for  maintaining vertex to vertex
        Set<Integer> set = new LinkedHashSet();
        int mcstCost = 0;
        while (set.size() <= this.V) {

            //for very first edge with min cost
            //put that in the set
            if (set.isEmpty()) {
                //fetching each edge connection in incr order
                Integer[] edgeAndCost = incrOrderOfEdgeList.get(0);
                int u = edgeAndCost[0];
                int v = edgeAndCost[1];
                int c = edgeAndCost[2];

                set.add(u);
                set.add(v);

                //once an edge is kept in set remove that from the incOrderEdgeList
                //so that we can access the same edge again and again
                incrOrderOfEdgeList.remove(0);
                //cost is incremented with the desired least costed edge-cost
                mcstCost += c;
                g.addVertexConnections(u, v, c, c);

            } else {
                //for every other new edge 
                int lesserC = Integer.MAX_VALUE;
                int edgeIndexToAdd = -1;
                Iterator itr = set.iterator();
                while (itr.hasNext()) {
                    //we will get each vertex in the set
                    //and check if that vertex is forming any edge with another vertex
                    //vertex either be (u_, vertex) or (vertex, v_)
                    int verInSet = (Integer) itr.next();
                    for (int k = 0; k < incrOrderOfEdgeList.size(); k++) {

                        //we are checking verInSet with all the edges in the list
                        Integer[] eac = incrOrderOfEdgeList.get(k);
                        int u_ = eac[0];
                        int v_ = eac[1];
                        int c_ = eac[2];

                        //this if is vertex either be (u_, vertex) or (vertex, v_)
                        if (verInSet == u_ || verInSet == v_) {

                            //this if check the new edge(u_, v_) that haas been founded
                            //both of them should not be in the set otherwise 
                            //it will create a cycle
                            //set = 1-2-3-4
                            //verInSet = 2
                            //u_, v_ = 4,2
                            //so here 2 is part of 4,2 but 4,2 are also in the set
                            //which is creating cycle so in that we will have to skip
                            //this edge 4,2 even if it is less in cost
                            if (set.contains(u_) && set.contains(v_)) {
                                //cycle
                                //new edge u,v is already in set
                            } else {
                                //if it u_,v_ is not creating cycle
                                //we will check if the cost(c_) with this u_,v_
                                //is less than the lesserC
                                //we will update lesserC with least one
                                //and mark index k at which edge u_,v_ is found in the incrOrderCostList
                                if (c_ < lesserC) {
                                    lesserC = c_;
                                    edgeIndexToAdd = k;
                                }

                            }

                        }

                    }

                }

                //Now till here we have visited all the vertex(verInSet) in set
                //with all the edges(u_, v_) in the list
                //and till here we have found the index location (edgeIndexToAdd)
                //whose edge can form edge to edge with vertex in the set
                //and this edge has least cost also
                Integer[] eacToAdd = incrOrderOfEdgeList.get(edgeIndexToAdd);
                set.add(eacToAdd[0]);
                set.add(eacToAdd[1]);
                //add up all the least cost
                mcstCost += lesserC;
                //also remove that edgeIndexToAdd index from the list
                //so that we don't process the same edge again and again
                incrOrderOfEdgeList.remove(edgeIndexToAdd);

                g.addVertexConnections(eacToAdd[0], eacToAdd[1], eacToAdd[2], eacToAdd[2]);

            }

            //since for a Minimum Cost Spanning Tree
            //edge E = V-1
            //and tree must contain all the vertex V with E edges
            //if all the vertex is kept in the set  and equals to V
            //we will stop
            if (set.size() == this.V) {
                break;
            }

        }

        System.out.println("\nMCST from prim's algo: " + mcstCost);
        return g;

    }

}
