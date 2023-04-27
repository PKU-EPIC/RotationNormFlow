import flow.rottrans as rt
import flow.squeezetrans as st


def get_affine(config, feature_dim, first_layer_condition=False):
    if first_layer_condition:
        if config.rot == '16UnTrans': ## 4x4 
            if config.lu:
                return st.Condition16TransLU(feature_dim)
            else:
                return st.Condition16Trans(feature_dim)
        elif config.rot == '16UnRot':
            return rt.ConditionRot(feature_dim)
    if config.condition:
        ### Affine function
        if config.rot == '16Trans': ## 4x4 
            if config.lu: ### with LU decomposition
                return st.Condition16TransLU(feature_dim)
            else:
                return st.Condition16Trans(feature_dim)
        if config.rot == '16UnTrans': ## 4x4 
            if config.lu: ### with LU decomposition
                return st.Uncondition16TransLU()
            else:
                return st.Uncondition16Trans()
        elif config.rot == '36Trans': ## 6x6
            return st.Condition36Trans(feature_dim)
        ### Ablation with "3x3 affine transformation"
        elif config.rot == '9TransLSVD':  ## 3x3 + SVD
            return rt.Condition9RotL(feature_dim)
        elif config.rot == '9TransRSVD':   ## 3x3 + SVD
            return rt.Condition9RotR(feature_dim)
        elif config.rot == '9TransLSmith':   ## 3x3 + Smith
            if config.lu:
                return st.Condition9TransLU(feature_dim)
            else:
                return st.Condition9Trans(feature_dim)
        elif config.rot == '9TransRSmith':  ## 3x3 + Smith
            return rt.Condition9RotRSmith(feature_dim)
        ### Pure Rotation function
        elif config.rot == '16Rot': ## SVD(4x4) -> 4x4 Rotation
            return rt.ConditionRot(feature_dim)
        elif config.rot == '16UnRot': ## SVD(4x4) -> 4x4 Rotation
            return rt.UnconditionRot()
        else:
            return None
        
    else:
        ### Affine function
        if config.rot == '16Trans': ## 4x4 
            if config.lu: ### with LU decomposition
                return st.Uncondition16TransLU()
            else:
                return st.Uncondition16Trans()
        elif config.rot == '36Trans': ## 6x6
            return st.Uncondition36Trans()
        ### Ablation with "3x3 affine transformation"
        elif config.rot == '9TransLSVD':  ## 3x3 + SVD
            return rt.Uncondition9RotL()
        elif config.rot == '9TransRSVD':   ## 3x3 + SVD
            return rt.Uncondition9RotR()
        elif config.rot == '9TransLSmith':   ## 3x3 + Smith
            if config.lu:
                return st.Uncondition9TransLU()
            else:
                return st.Uncondition9Trans()
        elif config.rot == '9TransRSmith':  ## 3x3 + Smith
            return rt.Uncondition9RotRSmith()
        ### Pure Rotation function
        elif config.rot == '16Rot': ## SVD(4x4) -> 4x4 Rotation
            return rt.UnconditionRot()
        else:
            return None
        