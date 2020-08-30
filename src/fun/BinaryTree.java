/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import javafx.util.Pair;

/**
 *
 * @author RAVI
 */
public class BinaryTree<T> {

    private TreeNode<T> root;
    private int height = 0;

    public BinaryTree() {
        this.root = null;
    }

    public BinaryTree(TreeNode rootNode) {
        root = rootNode;
    }

    public BinaryTree(T data) {
        root = new TreeNode(data, null, null);
    }

    public BinaryTree(T data, TreeNode left, TreeNode right) {
        root = new TreeNode(data, left, right);
    }

    public TreeNode<T> getRoot() {
        return root;
    }

    public void setRoot(TreeNode<T> root) {
        this.root = root;
    }

    public void insert(T data) {

        if (root == null) {
            this.root = new TreeNode(data);
            return;
        }

        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);

        while (!q.isEmpty()) {

            TreeNode t = q.poll();
            if (t.getLeft() == null) {
                t.setLeft(new TreeNode(data));
                return;
            } else if (t.getRight() == null) {
                t.setRight(new TreeNode(data));
                return;
            }

            q.add(t.getLeft());
            q.add(t.getRight());

        }

    }

    public void treeBFS() {

        //tree bfs is also level order
        if (root == null) {
            System.out.println("Tree is empty");
            return;
        }

        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);

        while (!q.isEmpty()) {

            TreeNode t = q.poll();
            System.out.print(t.toString() + " ");
            if (t.getLeft() != null) {
                q.add(t.getLeft());
            }

            if (t.getRight() != null) {
                q.add(t.getRight());
            }

        }

    }

    private void inorder(TreeNode node) {

        if (node == null) {
            return;
        }

        if (node.getLeft() != null) {
            inorder(node.getLeft());
        }

        System.out.print(node.toString() + " ");

        if (node.getRight() != null) {
            inorder(node.getRight());
        }

    }

    public void treeInorder() {

        if (root == null) {
            System.out.println("Tree is empty");
            return;
        } else {
            inorder(this.root);
        }

    }

    private void preorder(TreeNode node) {

        if (node == null) {
            return;
        }

        System.out.print(node.toString() + " ");

        if (node.getLeft() != null) {
            preorder(node.getLeft());
        }

        if (node.getRight() != null) {
            preorder(node.getRight());
        }

    }

    public void treePreorder() {

        if (root == null) {
            System.out.println("Tree is empty");
            return;
        } else {
            preorder(this.root);
        }

    }

    private void postorder(TreeNode node) {

        if (node == null) {
            return;
        }

        if (node.getLeft() != null) {
            postorder(node.getLeft());
        }

        if (node.getRight() != null) {
            postorder(node.getRight());
        }

        System.out.print(node.toString() + " ");

    }

    public void treePostorder() {

        if (root == null) {
            System.out.println("Tree is empty");
            return;
        } else {
            postorder(this.root);
        }

    }

    private void leftOuterBoundary_TopToBottom(TreeNode node, Set<T> set) {

        if (node == null) {
            return;
        }

        //System.out.print(node.toString()+" ");
        set.add((T) node.getData());
        if (node.getLeft() != null) {
            leftOuterBoundary_TopToBottom(node.getLeft(), set);
        }

    }

    private void rightOuterBoundary_BottomToTop(TreeNode node, Set<T> set) {

        if (node == null) {
            return;
        }

        //going down deep in right tree
        //print last of that root and recursive back up to top
        if (node.getRight() != null) {
            rightOuterBoundary_BottomToTop(node.getRight(), set);
        }
        //System.out.print(node.toString()+" ");
        set.add((T) node.getData());
    }

    private void leafLevelBoundary_LeftToRight(TreeNode node, Set<T> set) {

        if (node == null) {
            return;
        }

        if (node.getLeft() != null) {
            leafLevelBoundary_LeftToRight(node.getLeft(), set);
        }

        if (node.getLeft() == null && node.getRight() == null) {
            set.add((T) node.getData());
            //System.out.print(node.toString()+" ");
        }

        if (node.getRight() != null) {
            leafLevelBoundary_LeftToRight(node.getRight(), set);
        }

    }

    private void outerBoundary(Set<T> set) {

        leftOuterBoundary_TopToBottom(root, set);

        leafLevelBoundary_LeftToRight(root.getLeft(), set);
        leafLevelBoundary_LeftToRight(root.getRight(), set);

        rightOuterBoundary_BottomToTop(root, set);

    }

    public void treeOuterBoundry() {

        //from root to left then to leaf and back to root
        //in anticlockwise fasion
        if (root == null) {
            System.out.println("Tree is empty");
            return;
        } else {
            Set<T> set = new LinkedHashSet<>();
            outerBoundary(set);
            System.out.print(set.toString());
        }

    }

    private void zigzag(boolean side) {

        Stack<TreeNode> startStack = new Stack<>();
        Stack<TreeNode> internalStack = new Stack<>();

        startStack.push(root);
        boolean leftToRight = side;
        while (!startStack.isEmpty()) {

            TreeNode t = startStack.pop();

            System.out.print(t.toString() + " ");

            if (leftToRight) {

                if (t.getRight() != null) {
                    internalStack.push(t.getRight());
                }

                if (t.getLeft() != null) {
                    internalStack.push(t.getLeft());
                }

            } else {

                if (t.getLeft() != null) {
                    internalStack.push(t.getLeft());
                }

                if (t.getRight() != null) {
                    internalStack.push(t.getRight());
                }

            }

            if (startStack.isEmpty()) {

                leftToRight = !leftToRight;
                Stack<TreeNode> temp = startStack;
                startStack = internalStack;
                internalStack = temp;

            }

        }

    }

    public void treeZigZag(boolean leftToRight) {

        if (root == null) {
            System.out.println("Tree is empty");
            return;
        } else {
            zigzag(leftToRight);
        }

    }

    private void kSumPathAndDeleteOtherNode(TreeNode node, int k, int sum, TreeNode parent) {

        sum += (int) node.getData();

        if (node.getLeft() != null) {
            kSumPathAndDeleteOtherNode(node.getLeft(), k, sum, node);
        }

        //leaf nodes
        //System.out.println("|| sum: " + sum + " node " + (int) node.getData());
        if (sum >= k) {
            //to check other nodes also adjust sum here only
            sum -= (int) node.getData();
            //System.out.println("-- sum: " + sum + " node " + (int) node.getData());
            return;
        } else if (sum < k && (node.getLeft() == null && node.getRight() == null)) {
            //to check other nodes also adjust sum here only
            sum -= (int) node.getData();
            //System.out.println(".. sum: " + sum + " node " + (int) node.getData() + " parent " + (int) parent.getData());
            if (node == parent.getLeft()) {
                parent.setLeft(null);
                //System.out.println("hhh ");
            } else if (node == parent.getRight()) {
                parent.setRight(null);
                //System.out.println("ggg");
            }

            return;
        }

        if (node.getRight() != null) {
            kSumPathAndDeleteOtherNode(node.getRight(), k, sum, node);
        }

    }

    public void kSumPathOfTree(int k) throws Exception {

        if (getRoot() == null) {
            System.out.println("Tree is empty");
            return;
        }

        if (!(getRoot().getData() instanceof Integer)) {
            throw new Exception("K sum path can be implemented as Integer generic type");
        }

        kSumPathAndDeleteOtherNode(root, k, 0, null);

    }
    
    public int treeHeightWithNode(TreeNode node){
        
        if(node == null){
            return 0;
        }
        
        return Math.max(treeHeightWithNode(node.getLeft()), treeHeightWithNode(node.getRight())) + 1;
        
    }
    
    public int treeHeight(){
        if (getRoot() == null) {
            System.out.println("Tree is empty");
            return 0;
        }
        
        return treeHeightWithNode(getRoot());
        
    }

    private int inorderRootElementIndex(T[] inorder, T data, int l, int r) {
        int inorderRootIndex = -1;
        for (int i = l; i <= r; i++) {

            if (data == inorder[i]) {
                inorderRootIndex = i;
                break;
            }

        }

        return inorderRootIndex;
    }

//    private void buildTree(T[] inorder, T[] preorder, int l, int r, TreeNode<T> root, int kPreorderIndex) {
//
//        if (r < 0 || l < 0) {
//            System.out.println("ggg");
//            return;
//        }
//
//        if (l > inorder.length - 1 || kPreorderIndex > inorder.length - 1) {
//            return;
//        }
//
//        T rootElementPreorder = preorder[kPreorderIndex++];
//        int inorderRootIndex = inorderRootElementIndex(inorder, rootElementPreorder, l, r);
//        root.setLeft(new TreeNode(rootElementPreorder));
//        System.out.println("....l" + l + " r " + inorderRootIndex + " K " + rootElementPreorder);
//        buildTree(inorder, preorder, l, inorderRootIndex, root.getLeft(), kPreorderIndex);
//        System.out.println(".=.=l" + (inorderRootIndex) + " r " + r + " K " + rootElementPreorder);
//        root.setRight(new TreeNode(rootElementPreorder));
//        buildTree(inorder, preorder, inorderRootIndex, r, root.getRight(), inorderRootIndex);
//
//    }
    private TreeNode<T> buildTree(T[] inorder, T[] preorder, int l, int r, int preIndex) {

        if (l > r) {
            System.out.println("ggg");
            return null;
        }
        
        if(preIndex > preorder.length-1){
            return null;
        }

        TreeNode<T> node = new TreeNode<T>(preorder[preIndex++]);

        if (l == r) {
            System.out.println("-----l" + l + " r " + r + " K " + preIndex+" d "+node.getData());
            return node;
        }

        int inorderRootIndex = inorderRootElementIndex(inorder, node.getData(), l, r);
        System.out.println("....l" + l + " r " + inorderRootIndex + " K " + preIndex+" d "+node.getData());
        node.setLeft(buildTree(inorder, preorder, l, inorderRootIndex - 1, preIndex));
        System.out.println("||||l" + l + " r " + inorderRootIndex + " K " + preIndex+" d "+node.getData());       
        node.setRight(buildTree(inorder, preorder, inorderRootIndex + 1, r, preIndex));
        
        return node;

    }

    public BinaryTree<T> buildTreeFromInorderPreorder(T[] inorder, T[] preorder) {

        if (inorder.length != preorder.length) {
            throw new RuntimeException("Tree nodes are not equal");
        }

        int n = inorder.length;
        int preIndex = 0;
        TreeNode<T> root = buildTree(inorder, preorder, 0, n - 1, preIndex);

//        int n = inorder.length;
//        int kPreorderIndex = 0;
//        T rootElementPreorder = preorder[kPreorderIndex++];
//        int inorderRootIndex = inorderRootElementIndex(inorder, rootElementPreorder, 0, n);
//        
//        TreeNode<T> root = new TreeNode<>(rootElementPreorder);
//        System.out.println("--l" + (0)+" r "+inorderRootIndex+" K "+rootElementPreorder);
//        buildTree(inorder, preorder, 0, inorderRootIndex, root, kPreorderIndex);
//        System.out.println("!!!! l" + (inorderRootIndex+1)+" r "+n+" K "+preorder[inorderRootIndex+1]);
//        buildTree(inorder, preorder, inorderRootIndex+1, n, root, inorderRootIndex+1);
        return new BinaryTree<T>(root);

    }
    
    public void treeTopView(){
        //inner class scope limited to this method only
        class NodePair{
            TreeNode node;
            int horizontalDistance;

            public NodePair(TreeNode node, int horizontalDistance) {
                this.node = node;
                this.horizontalDistance = horizontalDistance;
            }
            
        }
        
        Queue<NodePair> q = new LinkedList<>();
        Map<Integer, TreeNode> map = new TreeMap<>();
        
        if(root == null){
            System.out.println("Tree is empty");
            return;
        }
        
        q.add(new NodePair(root, 0));
        
        while(!q.isEmpty()){
            
            NodePair t = q.poll();
            
            if(!map.containsKey(t.horizontalDistance)){
                map.put(t.horizontalDistance, t.node);
            }
            
            if(t.node.getLeft()!=null){
                q.add(new NodePair(t.node.getLeft(), t.horizontalDistance-1));
            }
            
            if(t.node.getRight()!=null){
                q.add(new NodePair(t.node.getRight(), t.horizontalDistance+1));
            }
            
        }
        
        for(Map.Entry<Integer, TreeNode> e: map.entrySet()){
            System.out.print(e.getValue().getData()+" ");
        }
        
        
    }
    
    public void treeInorderIterative(){
        
        if(root == null){
            System.out.println("Tree is empty");
            return;
        }
        
        Stack<Pair<TreeNode<T>, Integer>> st = new Stack<>();
        st.push(new Pair<>(getRoot(), 0));
        
        while(!st.isEmpty()){
            
            Pair<TreeNode<T>, Integer> p = st.pop();
            TreeNode<T> t = p.getKey();
            Integer status = p.getValue();
            
            if(t == null || status == 3){
                continue;
            }
            
            st.push(new Pair<>(t, status+1));
            
            if(status == 0) st.push(new Pair<>(t.getLeft(), 0));
            if(status == 1) System.out.print(t.getData()+" ");
            if(status == 2) st.push(new Pair<>(t.getRight(), 0));
            
        }
        
    }
    
    public void treePreorderIterative(){
        
        if(root == null){
            System.out.println("Tree is empty");
            return;
        }
        
        Stack<Pair<TreeNode<T>, Integer>> st = new Stack<>();
        st.push(new Pair<>(getRoot(), 0));
        
        while(!st.isEmpty()){
            
            Pair<TreeNode<T>, Integer> p = st.pop();
            TreeNode<T> t = p.getKey();
            Integer status = p.getValue();
            
            if(t == null || status == 3){
                continue;
            }
            
            st.push(new Pair<>(t, status+1));
            
            if(status == 0) System.out.print(t.getData()+" ");
            if(status == 1) st.push(new Pair<>(t.getLeft(), 0));
            if(status == 2) st.push(new Pair<>(t.getRight(), 0));
            
        }
        
    }
    
    public void treePostorderIterative(){
        
        if(root == null){
            System.out.println("Tree is empty");
            return;
        }
        
        Stack<Pair<TreeNode<T>, Integer>> st = new Stack<>();
        st.push(new Pair<>(getRoot(), 0));
        
        while(!st.isEmpty()){
            
            Pair<TreeNode<T>, Integer> p = st.pop();
            TreeNode<T> t = p.getKey();
            Integer status = p.getValue();
            
            if(t == null || status == 3){
                continue;
            }
            
            st.push(new Pair<>(t, status+1));
            
            if(status == 0) st.push(new Pair<>(t.getLeft(), 0));
            if(status == 1) st.push(new Pair<>(t.getRight(), 0));
            if(status == 2) System.out.print(t.getData()+" ");
            
        }
        
    }

}
