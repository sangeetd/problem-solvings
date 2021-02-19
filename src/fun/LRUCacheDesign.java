/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author sangeetdas
 */
public class LRUCacheDesign {

    private class Node {

        int key;
        int value;
        Node prev;
        Node next;

        public Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }

    private int capacity;
    Map<Integer, Node> map;
    Node head;
    Node tail;

    public LRUCacheDesign(int capacity) {
        this.capacity = capacity;
        map = new HashMap<>(capacity);
        head = null;
        tail = null;
    }

    public int get(int key) {
        if (!map.containsKey(key)) {
            return -1;
        }
        Node node = map.get(key);
        remove(node);
        setHead(node);

        return node.value;
    }

    public void put(int key, int value) {
        if (map.containsKey(key)) {
            Node node = map.get(key);
            node.value = value;
            remove(node);
            setHead(node);
        } else {
            if (map.size() >= capacity) {
                map.remove(tail.key);
                remove(tail);
            }
            Node node = new Node(key, value);
            setHead(node);
            map.put(key, node);
        }
    }

    private void remove(Node node) {
        //if node is first node of DLL
        if (node.prev == null) {
            //move head pointer to next of node
            head = node.next;
        } else {
            Node prevNode = node.prev;
            prevNode.next = node.next;
        }

        //if the node is last node of DLL
        if (node.next == null) {
            //update tail pointer to node's prev 
            tail = node.prev;
        } else {
            Node nextNode = node.next;
            nextNode.prev = node.prev;
        }
    }

    private void setHead(Node node) {
        node.prev = null;
        node.next = head;
        if (head != null) {
            // Null <- head -> ...
            head.prev = node;
            // node <- head -> ...
            
        }

        head = node;
        //head = Null <- node -> ... 
        if (tail == null) {
            tail = node;
        }
    }

}
