/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

/**
 *
 * @author RAVI
 */
public class BinarySearchTree<T> extends BinaryTree<T> implements Comparator<T> {

    public BinarySearchTree() {
    }

    public BinarySearchTree(T data) {
        super(data);
    }

    public BinarySearchTree(T data, TreeNode<T> left, TreeNode<T> right) {
        super(data, left, right);
    }

    public BinarySearchTree(TreeNode<T> rootNode) {
        super(rootNode);
    }

    @Override
    public void insert(T data) {
        insertBST(getRoot(), data);
    }

    private TreeNode<T> insertBST(TreeNode<T> node, T data) {

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
        } else if (compare((T) node.getData(), (T) data) >= 0) {
            TreeNode t = insertBST(node.getLeft(), data);
            node.setLeft(t);
        }

        return node;
    }
    
    private List<TreeNode<T>> deleteUtilFindInorderSuccessor(TreeNode<T> root){
        
        if(root == null){
            return Collections.emptyList();
        }
        
        TreeNode<T> succPrev = null;
        TreeNode<T> succ = null;
        if(root.getRight() != null){
            succPrev = root;
            succ = root.getRight(); //succ is left most node in the right sub-tree of the given root node
            while(succ.getLeft() != null){
                succPrev = succ;
                succ = succ.getLeft();
            }
        }
        return Arrays.asList(succPrev, succ);
    }
    
    private TreeNode<T> deleteUtil(TreeNode<T> rootToDelete){
        
        if(rootToDelete == null){
            return null;
        }
        
        if(rootToDelete.getLeft() == null && rootToDelete.getRight() == null){
            //leaf
            return null;
        }else if(rootToDelete.getLeft() == null){
            //has one child
            return rootToDelete.getRight();
        }else if(rootToDelete.getRight() == null){
            //has one child
            return rootToDelete.getLeft();
        }else {
            //has two children
            List<TreeNode<T>> succList = deleteUtilFindInorderSuccessor(rootToDelete);
            if(!succList.isEmpty()){
                
                TreeNode<T> succPrev = succList.get(0);
                TreeNode<T> succ = succList.get(1);
                
                //replace root's data with inorder succ's data
                rootToDelete.setData(succ.getData());
                //delete inorder succ from its actual place
                if(succPrev == rootToDelete){
                    succPrev.setRight(deleteUtil(succ));
                }else{
                    succPrev.setLeft(deleteUtil(succ));
                }
            }
            return rootToDelete;
        }
        
    }
    
    private TreeNode<T> deleteHelper(TreeNode<T> root, T findToDelete){
        if(root == null){
            return null;
        }
        
        if(root.getData() == findToDelete){
            return deleteUtil(root);
        }else if(compare(root.getData(), findToDelete) >= 0){
            root.setLeft(deleteHelper(root.getLeft(), findToDelete));
        }else{
            root.setRight(deleteHelper(root.getRight(), findToDelete));
        }
        return root;
    }
    
    public void delete(T findToDelete){
         deleteHelper(getRoot(), findToDelete);
    }

    @Override
    public int compare(T o1, T o2) {

        if (o1 instanceof Integer && o2 instanceof Integer) {
            Integer o11 = (Integer) o1;
            Integer o22 = (Integer) o2;

            return o11.compareTo(o22);
        }

        return -1;
    }

    private boolean isBSTUtil(TreeNode node, int min, int max) {

        if (node == null) {
            return true;
        }

        if ((Integer) node.getData() < min || (Integer) node.getData() > max) {
            return false;
        }

        return (isBSTUtil(node.getLeft(), min, (Integer) node.getData() - 1)
                && isBSTUtil(node.getRight(), (Integer) node.getData() + 1, max));

    }

    public boolean isBST() throws Exception {

        if (!(getRoot().getData() instanceof Integer)) {
            throw new Exception("Checking tree if BST or not is possbile for integer data type");
        }

        return isBSTUtil(getRoot(), Integer.MIN_VALUE, Integer.MAX_VALUE);
    }
}
