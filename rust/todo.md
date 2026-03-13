currently the python project dynamic_batch is a mess, largely because python is not strictly typed, and its also not possible to enforce boundaries

you will need to update flake.nix to add support for the rust toolchain, your env will not automatically update after that, so you need to run rust commands inside of a nix shell

so please rewrite the core part of it in rust, with support for being imported to python. to keep proper separation of concerns, we will have many rust crates
1. comm API bases
2. manager to runner comm API
3. primary to secondary manager comm API
4. scheduler api
these are all network agnostic!
now the network implementations:
1. networked over quic / wss 
2. thread-local via message passing
3. via sockets (both from parent to child, and fd sockets)
now the operations:
1. local manager impl
2. primary / secondary manager impl
3. runner impl
4. memory-constrained memory-stealing schuduler impl
others:
1. python provider 

all states in the protocol, and manager must be handled via ZST state machines, if there is a good crate for ZST state machines use it. 

everything in rust should be async, single threaded, and the async should be exposed to python!
the code should be written so that none but the python provider crate, has to know anything about python. thats why we require all of the api to assume single threaded async, so that we do not need send/sync support
always prefer async mpsc and oneshot over mutexes, if a method relies on receiving on multiple channels, we must select over all of them, as ordered listening can cause deadlocks.

for testing do not use SLURM rn, test everything locally


please add to your plan as a last step: continuously work on this. never end the session, add this to your memory now. once you have everything working, incrementally change the python dynamic_batch code the use the rust provided python package. at first every change should be config activated, so you can run both and compare. be aware that the python code has some major bugs though. 



DONE: the specific code that runs e.g. binary identifier is user dependent and the code here must be generic over it.
(BinaryIdentifier replaced with generic `I: Identifier` trait parameter throughout all crates.
 Concrete `TokenizerIdentifier` moved to `db_python_provider`.)
