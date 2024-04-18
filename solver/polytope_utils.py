import pypoman


def compute_RBPUA_Hrep(collide_input):

    relative_coordinates = collide_input[:,-2:] - collide_input[:,:2]
    relative_coordinates = relative_coordinates.view(relative_coordinates.shape[0], -1)
    relative_coordinates = relative_coordinates.detach().cpu().numpy()

    try:
        
        A, B = pypoman.compute_polytope_halfspaces(relative_coordinates)
    
    except Exception as e:
        
        A, B = [], []

    return A, B