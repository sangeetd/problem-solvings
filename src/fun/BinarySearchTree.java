/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.Comparator;
import java.util.LinkedList;
import java.util.Queue;

/**
 *
 * @author RAVI
 */
public class BinarySearchTree<T> extends BinaryTree implements Comparator<T> {

    public BinarySearchTree() {
    }

    public BinarySearchTree(Object data) {
        super(data);
    }

    public BinarySearchTree(Object data, TreeNode left, TreeNode right) {
        super(data, left, right);
    }

    @Override
    public void insert(Object data) {
        insertBST(getRoot(), data);
    }

    private TreeNode insertBST(TreeNode node, Object data) {

        if (getRoot() == null) {
            setRoot(new TreeNode(data));
            return null;
        }
        
        if (node == null) {
            return new TreeNode(data);
        }

        if (compare((T) node.getData(), (T) data) < 0) {
                //means data is greater than getData()
            //move to right sub tree
            TreeNode t = insertBST(node.getRight(), data);
            node.setRight(t);
        } else if (compare((T) node.getData(), (T) data) >= 0){
            TreeNode t = insertBST(node.getLeft(), data);
            node.setLeft(t);
        }
        
        return node;

    }

    @Override
    public int compare(T o1, T o2) {

        if (o1 instanceof Integer && o2 instanceof Integer) {
            Integer o11 = (Integer) o1;
            Integer o22 = (Integer) o2;

            return o11.intValue() - o22.intValue();

        }

        return -1;
    }
    
    private boolean isBSTUtil(TreeNode node, int min, int max) {
        
        if(node == null){
            return true;
        }
        
        if((Integer)node.getData()<min || (Integer)node.getData()>max){
            return false;
        }
        
        return (isBSTUtil(node.getLeft(), min, (Integer)node.getData()-1) && 
                isBSTUtil(node.getRight(), (Integer)node.getData()+1, max));
        
    }
    
    public boolean isBST() throws Exception{
        
        if(!(getRoot().getData() instanceof Integer)){
            throw new Exception("Checking tree if BST or not is possbile for integer data type");
        }
        
        return isBSTUtil(getRoot(), Integer.MIN_VALUE, Integer.MAX_VALUE);
    }
}
