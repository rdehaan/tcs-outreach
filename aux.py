from collections import defaultdict
import re

import networkx as nx
import matplotlib.pyplot as plt

import clingo
from clingox.reify import reify_program


def compute_graphs(
    program,
    verbose=True,
    num_models=0,
):
    control = clingo.Control(["--project", "-Wnone", "--heuristic=Domain"])
    control.add("base", [], program)
    if verbose:
        print(".. Grounding ..")
    control.ground([("base", [])])
    control.configuration.solve.models = num_models
    if verbose:
        print(".. Solving ..")
    graphs = []
    with control.solve(yield_=True) as handle:
        for model in handle:
            graph = [
                str(atom) for atom in model.symbols(atoms=True)
                if atom.name in ["edge", "node"]
            ]
            graphs.append(graph)
            if verbose:
                print(".", end='', flush=True)
        handle.get()
    if verbose:
        print("\n.. Done solving ..")
        print(f"Total solving time: {control.statistics['summary']['times']['solve']:.2f} sec")
    return graphs

def print_and_analyze_graph(
        graph
):
    print(f"GRAPH: {'. '.join(graph)}.")
    print(f" - Num nodes: {compute_num_nodes(graph)}")
    print(f" - Num edges: {compute_num_edges(graph)}")
    print(f" - Min VC size: {compute_min_vc_size(graph)}")
    print(f" - Min DS size: {compute_min_domset_size(graph)}")
    print(f" - Max IS size: {compute_max_indset_size(graph)}")
    print(f" - Min IDS size: {compute_min_inddomset_size(graph)}")
    print(f" - Has Hamiltonian cycle: {has_hamiltonian_cycle(graph)}")
    print(f" - Is 2-colorable: {is_two_colorable(graph)}")
    print(f" - Is 3-colorable: {is_three_colorable(graph)}")
    print(f" - Degree: {compute_degree(graph)}")
    display_graph(graph)


def compute_num_nodes(graph):
    program = """
        choose(X) :- node(X).
    """
    return compute_opt_size(graph, program)


def compute_num_edges(graph):
    program = """
        choose(X,Y) :- edge(X,Y), X < Y.
    """
    return compute_opt_size(graph, program)


def compute_min_vc_size(graph):
    program = """
        { choose(X) : node(X) }.
        :- edge(X,Y), not choose(X), not choose(Y).
        #minimize { 1,X : choose(X) }.
    """
    return compute_opt_size(graph, program)


def compute_min_domset_size(graph):
    program = """
        { choose(X) : node(X) }.
        :- node(X), not choose(X), not choose(Y) : edge(X,Y).
        #minimize { 1,X : choose(X) }.
    """
    return compute_opt_size(graph, program)


def compute_max_indset_size(graph):
    program = """
        { choose(X) : node(X) }.
        :- choose(X), choose(Y), edge(X,Y).
        #maximize { 1,X : choose(X) }.
    """
    return compute_opt_size(graph, program)


def compute_min_inddomset_size(graph):
    program = """
        { choose(X) : node(X) }.
        :- choose(X), choose(Y), edge(X,Y).
        :- node(X), not choose(X), not choose(Y) : edge(X,Y).
        #minimize { 1,X : choose(X) }.
    """
    return compute_opt_size(graph, program)


def has_hamiltonian_cycle(graph):
    program = """
        num_nodes(Num) :- Num = #count { N : node(N) }.
        h_num(1..N) :- num_nodes(N).
        1 { h_order(U,N) : h_num(N) } 1 :- node(U).
        1 { h_order(U,N) : node(U) } 1 :- h_num(N).
        :- num(N), num(N+1), h_order(U1,N), h_order(U2,N+1), not edge(U1,U2).
        :- h_order(U1,n), h_order(U2,1), not edge(U1,U2).
    """
    return has_property(graph, program)


def is_three_colorable(graph):
    program = """
        color(1..3).
        1 { color(X,C) : color(C) } 1 :- node(X).
        :- edge(X,Y), color(X,C), color(Y,C).
    """
    return has_property(graph, program)


def is_two_colorable(graph):
    program = """
        color(1..2).
        1 { color(X,C) : color(C) } 1 :- node(X).
        :- edge(X,Y), color(X,C), color(Y,C).
    """
    return has_property(graph, program)


def compute_opt_size(
    graph,
    program,
):
    base_program = """
        node(N) :- edge(N,_).
        edge(N,M) :- edge(M,N).
    """
    base_program += ".\n".join(graph) + ".\n"
    ctl = clingo.Control(["--project", "-Wnone", "--opt-mode=OptN"])
    ctl.add("base", [], program + base_program)
    ctl.ground([("base", [])])
    ctl.configuration.solve.models = 1
    opt_size = -1
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            choose_model = [
                str(atom) for atom in model.symbols(atoms=True)
                if atom.name in ["choose"]
            ]
            opt_size = len(choose_model)
    return opt_size


def has_property(
    graph,
    program,
):
    base_program = """
        node(N) :- edge(N,_).
        edge(N,M) :- edge(M,N).
    """
    base_program += ".\n".join(graph) + ".\n"
    ctl = clingo.Control(["--project", "-Wnone"])
    ctl.add("base", [], program + base_program)
    ctl.ground([("base", [])])
    ctl.configuration.solve.models = 1
    has_property = False
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            has_property = True
    return has_property

def compute_degree(graph):
    node_degree = defaultdict(lambda: 0)
    for atom in graph:
        if atom[:4] == "edge":
            nodes = re.findall(r'\d+', atom)
            node_degree[nodes[0]] += 1
    max_degree = max([node_degree[node] for node in node_degree])
    return max_degree

def display_graph(graph):
    nodes = []
    edges = []
    for atom in graph:
        if atom[:4] == "node":
            nodes.append(re.findall(r'\d+', atom)[0])
        if atom[:4] == "edge":
            nums = re.findall(r'\d+', atom)
            edges.append((nums[0], nums[1]))

    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.figsize"] = (8, 8)
    graph = nx.Graph()

    ax = plt.gca()
    ax.margins(0.08)

    options = {
        "edgecolors": "tab:gray",
        "node_size": 1000,
        "font_color": "white",
        "font_size": 10,
    }
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    layout = nx.planar_layout(graph)

    nx.draw(
        graph,
        layout,
        ax=ax,
        with_labels=True,
        **options
    )

    plt.show()


def construct_program(
    gen_program,
    max_indset_lower = None,
    max_indset_upper = None,
    min_domset_size = None,
    sym_breaking = False,
    require_3col = True,
    require_ham_cycle = True,
    require_planar = True,
    forbid_2col = False,
    require_ids_larger_than_ds = False,
):
    program = gen_program

    # Forbid isolated nodes
    program += """
        :- node(X), not edge(X,Y) : node(Y), X != Y.
    """
    # Ensure that the graph is connected
    program += """
        reachable(1).
        reachable(X) :- node(X), reachable(Y), edge(X,Y).
        :- node(X), not reachable(X).
    """

    if sym_breaking:
        # Symmetry breaking: order nodes by degree
        program += """
            degree(X,D) :- node(X), D = #count { Y : node(Y), edge(X,Y) }.
            :- node(X), node(Y), degree(X,DX), degree(Y,DY),
                X < Y, DX > DY.
        """

    if require_3col:
        # Require that the graph is 3-colorable
        program += """
            color(1..3).
            1 { color(X,C) : color(C) } 1 :- node(X).
            :- edge(X,Y), color(X,C), color(Y,C).
        """

    if require_ham_cycle:
        # Require that the graph has a Hamiltonian cycle
        program += """
            h_num(1..n).
            1 { h_order(U,N) : h_num(N) } 1 :- node(U).
            1 { h_order(U,N) : node(U) } 1 :- h_num(N).
            :- num(N), num(N+1), h_order(U1,N), h_order(U2,N+1), not edge(U1,U2).
            :- h_order(U1,n), h_order(U2,1), not edge(U1,U2).
            :- h_order(U2,n-1), h_order(U1,2), U1 > U2.
        """

    if max_indset_lower:
        # Require that there's an independent set of size max_indset_lower
        program += f"""
            {max_indset_lower} {{ indset(X) : node(X) }} {max_indset_lower}.
            :- indset(X), indset(Y), edge(X,Y).
        """

    if min_domset_size:
        # Require that there's a dominating set of size min_domset_size
        program += f"""
            {min_domset_size} {{ domset(X) : node(X) }} {min_domset_size}.
            :- node(X), not domset(X), not domset(Y) : edge(X,Y).
        """

    if require_planar:
        # Encode Felsner's planarity condition
        # (See, e.g., https://page.math.tu-berlin.de/~scheuch/publ/SAT2023-smsplanar.pdf)
        program += """
            order(1..3).
            num(1..n).
            1 { ordering(I,U,N) : num(N) } 1 :- order(I), node(U).
            1 { ordering(I,U,N) : node(U) } 1 :- order(I), num(N).
            triple_to_check(U,V,W) :- node(U), node(V), node(W), edge(U,V),
                U != V, U != W, V != W.
            1 { triple_order(U,V,W,I) : order(I) } 1 :- triple_to_check(U,V,W).
            :- triple_to_check(U,V,W), triple_order(U,V,W,I),
                ordering(I,U,NU),
                ordering(I,V,NV),
                ordering(I,W,NW),
                NU > NW.
            :- triple_to_check(U,V,W), triple_order(U,V,W,I),
                ordering(I,U,NU),
                ordering(I,V,NV),
                ordering(I,W,NW),
                NV > NW.
        """

    forbidden_base_program = gen_program

    forbidden_programs = []

    if forbid_2col:
        # Forbid that the graph is 2-colorable
        forbidden_programs.append("""
            color(1..2).
            1 { color(X,C) : color(C) } 1 :- node(X).
            :- edge(X,Y), color(X,C), color(Y,C).
        """)

    if max_indset_upper:
        # Forbid clique of size max_indset_upper+1
        forbidden_programs.append(f"""
            {max_indset_upper+1} {{ alt_indset(X) : node(X) }} {max_indset_upper+1}.
            :- alt_indset(X), alt_indset(Y), edge(X,Y).
        """)

    if min_domset_size:
        # Forbid DS of size min_domset_size-1
        forbidden_programs.append(f"""
        {min_domset_size-1} {{ alt_domset(X) : node(X) }} {min_domset_size-1}.
        :- node(X), not alt_domset(X), not alt_domset(Y) : edge(X,Y).
        """)

    if require_ids_larger_than_ds and min_domset_size:
        # Forbid IDS of size min_domset_size
        forbidden_programs.append(f"""
            {min_domset_size} {{ alt_inddomset(X) : node(X) }} {min_domset_size}.
            :- node(X), not alt_inddomset(X), not alt_inddomset(Y) : edge(X,Y).
            :- alt_inddomset(X), alt_inddomset(Y), edge(X,Y).
        """)

    if len(forbidden_programs) > 0:
        reified_program = ""
        for i, forbidden_program in enumerate(forbidden_programs):
            reified_symbols = reify_program(
                forbidden_base_program + forbidden_program,
                calculate_sccs=True,
            )
            reified_program += "".join([
                f"fprog({i},{symbol}).\n"
                for symbol in reified_symbols
            ])
            reified_program += f"fprog({i}).\n"

        # Modified version of:
        # https://github.com/potassco/clingo/blob/master/examples/reify/common/metaD.lp
        interpreter_program = """
            % NOTE: assumes that a rule has no more than one head

            fprog(I,sum(B,G,T)) :- fprog(I,rule(_,sum(B,G))), T = #sum { W,L : fprog(I,weighted_literal_tuple(B,L,W)) }.

            % extract supports of atoms and facts

            fprog(I,supp(A,B)) :- fprog(I,rule(     choice(H),B)), fprog(I,atom_tuple(H,A)).
            fprog(I,supp(A,B)) :- fprog(I,rule(disjunction(H),B)), fprog(I,atom_tuple(H,A)).

            fprog(I,supp(A)) :- fprog(I,supp(A,_)).

            fprog(I,atom(|L|)) :- fprog(I,weighted_literal_tuple(_,L,_)).
            fprog(I,atom(|L|)) :- fprog(I,literal_tuple(_,L)).
            fprog(I,atom( A )) :- fprog(I,atom_tuple(_,A)).

            fprog(I,fact(A)) :- fprog(I,rule(disjunction(H),normal(B))),
                fprog(I,atom_tuple(H,A)), not fprog(I,literal_tuple(B,_)).

            % generate interpretation

            fprog(I,true(atom(A)))                         :- fprog(I,fact(A)).
            fprog(I,true(atom(A))); fprog(I,fail(atom(A))) :- fprog(I,supp(A)), not fprog(I,fact(A)).
                           fprog(I,fail(atom(A)))          :- fprog(I,atom(A)), not fprog(I,supp(A)).

            fprog(I,true(normal(B))) :- fprog(I,literal_tuple(B)),
                fprog(I,true(atom(L))) : fprog(I,literal_tuple(B, L)), L > 0;
                fprog(I,fail(atom(L))) : fprog(I,literal_tuple(B,-L)), L > 0.
            fprog(I,fail(normal(B))) :- fprog(I,literal_tuple(B, L)), fprog(I,fail(atom(L))), L > 0.
            fprog(I,fail(normal(B))) :- fprog(I,literal_tuple(B,-L)), fprog(I,true(atom(L))), L > 0.

            fprog(I,true(sum(B,G))) :- fprog(I,sum(B,G,T)),
                #sum { W,L : fprog(I,true(atom(L))), fprog(I,weighted_literal_tuple(B, L,W)), L > 0 ;
                       W,L : fprog(I,fail(atom(L))), fprog(I,weighted_literal_tuple(B,-L,W)), L > 0 } >= G.
            fprog(I,fail(sum(B,G))) :- fprog(I,sum(B,G,T)),
                #sum { W,L : fprog(I,fail(atom(L))), fprog(I,weighted_literal_tuple(B, L,W)), L > 0 ;
                       W,L : fprog(I,true(atom(L))), fprog(I,weighted_literal_tuple(B,-L,W)), L > 0 } >= T-G+1.

            % verify supported model properties

            fprog(I,bot) :- fprog(I,rule(disjunction(H),B)), fprog(I,true(B)), fprog(I,fail(atom(A))) : fprog(I,atom_tuple(H,A)).
            fprog(I,bot) :- fprog(I,true(atom(A))), fprog(I,fail(B)) : fprog(I,supp(A,B)).

            % verify acyclic derivability

            fprog(I,internal(C,normal(B))) :- fprog(I,scc(C,A)), fprog(I,supp(A,normal(B))),
                fprog(I,scc(C,A')), fprog(I,literal_tuple(B,A')).
            fprog(I,internal(C,sum(B,G)))  :- fprog(I,scc(C,A)), fprog(I,supp(A,sum(B,G))),
                fprog(I,scc(C,A')), fprog(I,weighted_literal_tuple(B,A',W)).

            fprog(I,external(C,normal(B))) :- fprog(I,scc(C,A)), fprog(I,supp(A,normal(B))), not fprog(I,internal(C,normal(B))).
            fprog(I,external(C,sum(B,G)))  :- fprog(I,scc(C,A)), fprog(I,supp(A,sum(B,G))),  not fprog(I,internal(C,sum(B,G))).

            fprog(I,steps(C,Z-1)) :- fprog(I,scc(C,_)), Z = { fprog(I,scc(C,A)) : not fprog(I,fact(A)) }.

            fprog(I,wait(C,atom(A),0))   :- fprog(I,scc(C,A)), fprog(I,fail(B)) : fprog(I,external(C,B)), fprog(I,supp(A,B)).
            fprog(I,wait(C,normal(B),I)) :- fprog(I,internal(C,normal(B))), fprog(I,fail(normal(B))), fprog(I,steps(C,Z)), I = 0..Z-1.
            fprog(I,wait(C,normal(B),I)) :- fprog(I,internal(C,normal(B))), fprog(I,literal_tuple(B,A)),
                fprog(I,wait(C,atom(A),I)), fprog(I,steps(C,Z)), I < Z.
            fprog(I,wait(C,sum(B,G),I))  :- fprog(I,internal(C,sum(B,G))), fprog(I,steps(C,Z)), I = 0..Z-1, fprog(I,sum(B,G,T)),
                #sum { W,L :   fprog(I,fail(atom(L))),   fprog(I,weighted_literal_tuple(B, L,W)), L > 0, not fprog(I,scc(C,L)) ;
                       W,L : fprog(I,wait(C,atom(L),I)), fprog(I,weighted_literal_tuple(B, L,W)), L > 0,     fprog(I,scc(C,L)) ;
                       W,L :   fprog(I,true(atom(L))),   fprog(I,weighted_literal_tuple(B,-L,W)), L > 0                        } >= T-G+1.
            fprog(I,wait(C,atom(A),I))   :- fprog(I,wait(C,atom(A),0)), fprog(I,steps(C,Z)), I = 1..Z,
                fprog(I,wait(C,B,I-1)) : fprog(I,supp(A,B)), fprog(I,internal(C,B)).

            fprog(I,bot) :- fprog(I,scc(C,A)), fprog(I,true(atom(A))), fprog(I,wait(C,atom(A),Z)), fprog(I,steps(C,Z)).

            % saturate interpretations that are not answer sets

            fprog(I,true(atom(A))) :- fprog(I,supp(A)), not fprog(I,fact(A)), fprog(I,bot).
            fprog(I,fail(atom(A))) :- fprog(I,supp(A)), not fprog(I,fact(A)), fprog(I,bot).
        """

        glue_program = """
            :- not fprog(I,bot), fprog(I).
            fprog(I,bot) :- edge(X,Y), fprog(I,output(edge(X,Y),B)), fprog(I,fail(normal(B))), fprog(I).
            fprog(I,bot) :- not edge(X,Y), fprog(I,output(edge(X,Y),B)), fprog(I,true(normal(B))), fprog(I).
        """

        program += reified_program
        program += interpreter_program
        program += glue_program

    return program
