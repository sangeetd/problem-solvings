/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 *
 * @author RAVI
 */
public class Trie {

    private TrieNode root;

    public Trie() {
        root = new TrieNode('*');
    }

    public void insert(String str) {

        TrieNode temp = root;
        for (Character c : str.toCharArray()) {
            if (temp.charNode().containsKey(c)) {
                temp = temp.charNode().get(c);

            } else {
                TrieNode t = new TrieNode();
                temp.charNode().put(c, t);
                temp = t;
            }

        }

        temp.setEndOfWord(true);
        temp.increaseWordCount();

    }

    public boolean query(String str) {

        boolean isInDictionary = true;
        TrieNode temp = root;
        StringBuilder sb = new StringBuilder();
        for (Character c : str.toCharArray()) {

            if (temp.charNode().containsKey(c)) {

                sb.append(c);
                temp = temp.charNode().get(c);

            } else {
                isInDictionary = false;
                break;
            }

        }

        if (isInDictionary) {
            System.out.println("matched upto: " + sb.toString());
        } else {
            System.out.println("Unable to find word in your dictionary that start with " + str);
        }

        return isInDictionary;

    }

    private void wordsInDictionary(TrieNode n, StringBuilder str, Set<String> set) {

        for (Map.Entry<Character, TrieNode> e : n.charNode().entrySet()) {
            str.append(e.getKey());
            if (e.getValue().isEndOfWord() && e.getValue().charNode().size() >= 1) {
                set.add(str.toString());
                wordsInDictionary(e.getValue(), str, set);
            } else if (e.getValue().charNode().size() >= 1) {
                wordsInDictionary(e.getValue(), str, set);
            } else if (e.getValue().isEndOfWord()) {
                set.add(str.toString());
            }
            str.deleteCharAt(str.length() - 1);
        }

    }

    public Set<String> autoSuggestQuery(String str) {

        boolean isInDictionary = true;
        TrieNode temp = root;
        Set<String> autoSuggest = new HashSet<>();
        for (char c : str.toCharArray()) {

            if (temp.charNode().containsKey(c)) {
                temp = temp.charNode().get(c);
            } else {
                isInDictionary = false;
                break;
            }

        }

        if (temp.isEndOfWord()) {
            autoSuggest.add(str);
        }

        if (isInDictionary) {
            wordsInDictionary(temp, new StringBuilder(str), autoSuggest);
            System.out.println(autoSuggest.toString());
        } else {
            System.out.println("Unable to find word in your dictionary that start with " + str);
        }

        return autoSuggest;

    }

    public void wordCount(String str) {

        boolean isInDictionary = true;
        TrieNode temp = root;
        for (Character c : str.toCharArray()) {

            if (temp.charNode().containsKey(c)) {

                temp = temp.charNode().get(c);

            } else {
                isInDictionary = false;
                break;
            }

        }

        if (isInDictionary) {
            System.out.println(str + " word ocuurence is " + temp.getWordCount());
        } else {
            System.out.println("Unable to find word in your dictionary that start with " + str);
        }

    }

    public void delete(String str) {

        boolean isInDictionary = true;
        TrieNode temp = root;
        int prevChar = -1;
        TrieNode prevEOW = null;
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            if (temp.charNode().containsKey(c)) {
                temp = temp.charNode().get(c);

                //check if each char node is marked as EOW
//                System.out.println(c+" "+temp.charNode().size()+" "+temp.isEndOfWord());
                if (temp.isEndOfWord() && temp.charNode().size()>=1) {
                    //last occurence of EoW node is actually a word 
                    //in itself
                    prevEOW = temp;

                    //this prevChar+1 is what should be removed from
                    //prevEOW. becoz prevEOW is last independent word
                    //ex lets say words be peek, peekock
                    //you want to del peekock but peek is last independent word
                    //so peek should not be deleted p-e-e-k-EOW
                    //prevEOW = k-|o|-c-k = k's charNode that marks EoW for peek and holds o in peekock
                    //prevChar = k ; prevChar+1 = o
                    //so o should be removed.
                    prevChar = i;
                    
//                    System.out.println(str.charAt(i) +" "+i);
                    
                }
                
                

            } else {
                isInDictionary = false;
                break;
            }
            
        }
        
//        System.out.println(str.charAt(prevChar));

        if (isInDictionary && temp.isEndOfWord() && temp.charNode().size() >= 1) {
            //if you have word like geek and geekok
            //you want to del geek g-e-e-k-EOW(true)
            //notice you can't delete g-e-e-k-o
            //becz geekok contains sub part geek in it
            //so just maintain g-e-e-k-EOW(false)
            //this way logic behind finding a word on basis of EOW flag will
            //not be found for word geek anymore but only for geekok-EOW
            temp.setEndOfWord(false);
            temp.decreaseWordCount();
            System.out.println(str + " is deleted");
        } else if (isInDictionary && prevEOW != null && prevChar != -1) {
            //if we found intermedite sub word of a bigger word
            //like hill is sub word of hillktop
            //and you want to del hillktop then hill should not be affected
            //so h-i-l-l-prevEOW-remove(prevChar+1) = h-i-l-l-prevEOW-remove(k) 
            prevEOW.charNode().remove(str.charAt(prevChar + 1));
            System.out.println(str + " is deleted");
        } else if (isInDictionary && temp.isEndOfWord() && temp.charNode().size() <= 0 && prevEOW == null) {
            //if you have only word  hillktop, hello,hell
            //you want to del hillktop
            //becz hillktop contains no sub part so prevEOW will be null
            //temp will point to last p charNode trieNode value from map which is EOW
            //whose map will not contain any further node so size = 0
            deleteSpecial(str);
            System.out.println(str + " is deleted");
        } else {
            System.out.println("Unable to find word in your dictionary that start with " + str);
        }

    }

    private void deleteSpecial(String str) {

        Character prevParentChar = '\0';
        TrieNode prevParentCharNode = null;
        TrieNode temp = root;
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            if (temp.charNode().containsKey(c)) {
                if (temp.charNode().get(c).charNode().size() >= 2) {
                    //char c is dependent parent of 2 or more word
                    //let say hello, hell, hill, hillktop
                    //so h is parent to e, i
                    //just move forward
                    prevParentChar = c;
                    prevParentCharNode = temp;
                    temp = temp.charNode().get(c);
                } else {
                    prevParentCharNode.charNode().get(prevParentChar).charNode().remove(c);
                    break;
                }
            }

        }

    }

    public boolean update(String oldWord, String newWord) {

        if (query(oldWord)) {

            delete(oldWord);
            insert(newWord);

            return true;

        } else {
            System.out.println("old word is not found in the dictionary");
            return false;
        }

    }

    public void print() {

        Set<String> words = new HashSet<>();
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<Character, TrieNode> e : root.charNode().entrySet()) {

            if (e.getKey() != '*') {
                if(e.getValue().isEndOfWord()){
                 words.add(e.getKey().toString());   
                }
                wordsInDictionary(e.getValue(), sb.append(e.getKey()), words);
            }

            sb.setLength(0);

        }

        System.out.println(words.toString());

    }

}
