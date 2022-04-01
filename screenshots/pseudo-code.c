void Application()
{
    code_ptx = lireFichierPTX("mon_shader.ptx");

    // = Programme qui lancera potentiellement plusieurs millions de rayons sur le GPU
    // Ce aussi sera le point d'entrée du programme GPU
    // Ce n'est pas le programme de raytracing, mais celui que génerera le raytracing!
    raygen = compilerRayGen(code_ptx);

    // = Comment détecter les collisions avec la scène ?
    collision = compilerDetectionCollision(code_ptx);

    // = Que faire quand une collision est détectée ?
    onHit = compilerOnHit(code_ptx);
    
    bvh = creerStructureAcceleratrice(scene);
    
    pipeline = creerPipeline(raygen, collision, onHit);
    
    sbt = creerSBT()
    sbt.raygen = <Paramètres pour le programme raygen>;
    sbt.collision = <Paramètres globaux pour le programme de détection de collisions>;
    sbt.onHit = <Paramètres globaux pour le programme onHit>;
    
    pipeline = creerPipeline(sbt, raygen, collision, onHit);

    params = {} // Paramètres envoyés au raygen
    params.pixels = <Tableau 2D des pixels>;
    params.width = ...;
    params.height = ...;

    while(applicationOuverte)
    {
        lancerProgramme(pipeline, sbt, params);
        afficherTexture(params.pixels);
    }

    // On gère soi-même la mémoire, donc elle devra être libérée à la fin
    libererMémoire();
}

